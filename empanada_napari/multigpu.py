import os
import zarr
import yaml
import torch
import torch.hub
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from mitonet.inference import filters
from mitonet.inference.postprocess import factor_pad
from mitonet.inference.engines import MultiScaleInferenceEngine
from mitonet.inference.tracker import InstanceTracker
from mitonet.inference.matcher import SequentialMatcher
from mitonet.inference.array_utils import *
from mitonet.zarr_utils import *
from mitonet.aggregation.consensus import merge_objects3d

from empanada_napari.utils import ArrayData, resize, adjust_shape_by_scale


from tqdm import tqdm

from napari.qt.threading import thread_worker

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pickle
import functools

from collections import deque

from time import time

from mitonet.inference.postprocess import merge_semantic_and_instance

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD
            
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
            
def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    # all tensors are same size
    world_size = dist.get_world_size()
    max_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in range(world_size)
    ]
    
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for tensor in tensor_list:
        buffer = tensor.cpu().numpy().tobytes()
        data_list.append(pickle.loads(buffer))

    return data_list

def get_panoptic_seg(sem, instance_cells, config):
    label_divisor = config['engine_params']['label_divisor']
    thing_list = config['engine_params']['thing_list']
    stuff_area = config['engine_params']['stuff_area']
    void_label = config['engine_params']['void_label']
    
    # keep only label for instance classes
    instance_seg = torch.zeros_like(sem)
    for thing_class in thing_list:
        instance_seg[sem == thing_class] = 1

    # map object ids
    instance_seg = (instance_seg * instance_cells).long()

    pan_seg = merge_semantic_and_instance(
        sem, instance_seg, label_divisor, thing_list,
        stuff_area, void_label
    )
    
    return pan_seg

def run_forward_matchers(stack, axis, matchers, queue, config):
    # create the deques for sem and instance cells\
    confidence_thr = config['engine_params']['confidence_thr']
    median_kernel_size = config['engine_params']['median_kernel_size']
    mid_idx = (median_kernel_size - 1) // 2
    
    fill_index = 0
    sem_queue = deque(maxlen=median_kernel_size)
    cell_queue = deque(maxlen=median_kernel_size)

    while True:
        sem, cells = queue.get()
        
        if sem is None:
            break
        
        sem_queue.append(sem)
        cell_queue.append(cells)
        
        nq = len(sem_queue)
        if nq <= mid_idx:
            # take last item in the queue
            median_sem = sem_queue[-1]
            cells = cell_queue[-1]
        elif nq > mid_idx and nq < median_kernel_size:
            # continue while the queue builds
            median_sem = None
        else: 
            # nq == median_kernel_size
            # use the middle item in the queue
            # with the median segmentation probs
            median_sem = torch.median(
                torch.cat(list(sem_queue), dim=0), dim=0, keepdim=True
            ).values
            cells = cell_queue[mid_idx]
            
        # harden the segmentation to (N, 1, H, W)
        if median_sem is not None:
            if median_sem.size(1) > 1: # multiclass segmentation
                median_sem = torch.argmax(median_sem, dim=1, keepdim=True)
            else:
                median_sem = (median_sem >= confidence_thr).long() # need integers

            pan_seg = get_panoptic_seg(median_sem, cells, config)
        else:
            pan_seg = None
        
        if pan_seg is None:
            continue
        else:
            pan_seg = pan_seg.squeeze().numpy()
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)
                    
            zarr_put3d(stack, fill_index, pan_seg, axis)
            fill_index += 1
            
    # fill out the final segmentations
    for sem,cells in zip(list(sem_queue)[mid_idx + 1:], list(cell_queue)[mid_idx + 1:]):
        if sem.size(1) > 1: # multiclass segmentation
            sem = torch.argmax(sem, dim=1, keepdim=True)
        else:
            sem = (sem >= confidence_thr).long() # need integers

        pan_seg = get_panoptic_seg(sem, cells, config)
        pan_seg = pan_seg.squeeze().numpy()
        
        for matcher in matchers:
            if matcher.target_seg is None:
                pan_seg = matcher.initialize_target(pan_seg)
            else:
                pan_seg = matcher(pan_seg)
    
        zarr_put3d(stack, fill_index, pan_seg, axis)
        fill_index += 1
        
    return None

def main_worker(gpu, config, axis_name, volume, data_store):
    config['gpu'] = gpu
    rank = gpu
    axis = config['axes'][axis_name]
    
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001',
                            world_size=config['world_size'], rank=rank)
    
    
    # load models and set engine class
    base_model = torch.hub.load_state_dict_from_url(config['base_model_url'])
    render_model = torch.hub.load_state_dict_from_url(config['render_model_url'])
    engine_cls = MultiScaleInferenceEngine
    
    torch.cuda.set_device(config['gpu'])
    
    eval_tfs = A.Compose([
        A.Normalize(**config['norms']),
        ToTensorV2()
    ])
    
    # create the dataloader
    shape = volume.shape
    dataset = ArrayData(volume, config['engine_params']['inference_scale'], axis, eval_tfs)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True,
        drop_last=False, num_workers=0, sampler=sampler
    )
    
    # if in main process, create zarr to store results
    # chunk in axis direction only
    if rank == 0:
        chunks = [None, None, None]
        chunks[axis] = 1
        chunks = tuple(chunks)
        stack = data_store.create_dataset(f'panoptic_{axis_name}', shape=shape, 
                                          dtype=np.uint64, chunks=chunks, 
                                          overwrite=True)
       
        thing_list = config['engine_params']['thing_list']
        label_divisor = config['engine_params']['label_divisor']
        matchers = [
            SequentialMatcher(thing_class, **config['matcher_params'])
            for thing_class in thing_list
        ]
        
        queue = mp.Queue()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(stack, axis, matchers, queue, config))
        matcher_proc.start()
        
    # create the inference engine
    inference_engine = engine_cls(base_model, render_model, **config['engine_params'], device=f'cuda:{gpu}')
    inference_engine.base_model = DDP(inference_engine.base_model, device_ids=[config['gpu']])
    inference_engine.render_model = DDP(inference_engine.render_model, device_ids=[config['gpu']])
    
    print('Created inference engine on', inference_engine.device)
    
    scale_factor = config['engine_params']['inference_scale'] * 4
    
    iterator = dataloader if rank != 0 else tqdm(dataloader, total=len(dataloader))
    
    n = 0
    step = get_world_size()
    total_len = volume.shape[axis]
    
    for batch in iterator:
        index = batch['index']
        image = batch['image']
        h, w = image.size()[2:]
        image = factor_pad(image, config['padding_factor'])

        output = inference_engine.infer(image)
        instance_cells = inference_engine.get_instance_cells(
            output['ctr_hmp'], output['offsets']
        )
        
        # correctly resize the sem and instance_cells
        coarse_sem_logits = output['sem_logits']
        sem_logits = coarse_sem_logits.clone()
        features = output['semantic_x']
        sem, _, instance_cells = inference_engine.upsample_logits_and_cells(
            sem_logits, coarse_sem_logits, features, instance_cells, scale_factor
        )
        
        # get median semantic seg
        sems = all_gather(sem)
        instance_cells = all_gather(instance_cells)
        
        # drop last segs if unnecessary
        n += len(sems)
        stop = min(step, (total_len - n) + step)
        
        # move both sem and instance_cells to cpu
        sems = [sem.cpu() for sem in sems[:stop]]
        instance_cells = [cells.cpu() for cells in instance_cells[:stop]]
        
        if rank == 0:
            for sem, cells in zip(sems, instance_cells):
                queue.put(
                    (sem[..., :h, :w], cells[..., :h, :w])
                )
            
    # pass None to queue to mark the end of inference
    if rank == 0:
        queue.put((None, None))
        matcher_proc.join()

class MultiGPUOrthoplaneEngine:
    def __init__(
        self,
        store_url,
        model_config,
        inference_scale=1,
        label_divisor=1000,
        median_kernel_size=5,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.3,
        merge_iou_thr=0.25,
        merge_ioa_thr=0.25,
        force_connected=True,
        use_gpu=True,
        filters=None
    ):
        if store_url is not None:
            self.data = zarr.open(store_url, mode='w')
        else:
            self.data = None

        # check whether GPU is available
        if not torch.cuda.device_count() > 1:
            raise Exception(f'MultiGPU inference requires access to multiple GPUs!')
            
        device = 'gpu'
            
        # load the base and render models
        self.labels = model_config['labels']
        self.config = model_config
        self.config['base_model_url'] = model_config[f'base_model_{device}']
        self.config['render_model_url'] = model_config[f'render_model_{device}']
        
        self.config['engine_params'] = {}
        self.config['engine_params']['thing_list'] = self.config['thing_list']
        self.config['engine_params']['inference_scale'] = inference_scale
        self.config['engine_params']['label_divisor'] = label_divisor
        self.config['engine_params']['median_kernel_size'] = median_kernel_size
        self.config['engine_params']['stuff_area'] = stuff_area
        self.config['engine_params']['void_label'] = void_label
        self.config['engine_params']['nms_threshold'] = nms_threshold
        self.config['engine_params']['nms_kernel'] = nms_kernel
        self.config['engine_params']['confidence_thr'] = confidence_thr

        self.config['axes'] = {'xy': 0, 'xz': 1, 'yz': 2}
        self.config['matcher_params'] = {}
        
        self.config['matcher_params']['label_divisor'] = label_divisor
        self.config['matcher_params']['merge_iou_thr'] = merge_iou_thr
        self.config['matcher_params']['merge_ioa_thr'] = merge_ioa_thr
        self.config['matcher_params']['force_connected'] = force_connected
        
        self.config['world_size'] = torch.cuda.device_count()

    def update_params(
        self,
        store_url,
        inference_scale,
        label_divisor,
        median_kernel_size,
        nms_threshold,
        nms_kernel,
        confidence_thr,
        merge_iou_thr,
        merge_ioa_thr
    ):

        if store_url is not None:
            self.data = zarr.open(store_url, mode='w')
        else:
            self.data = None

        self.label_divisor = label_divisor
        self.merge_iou_thr = merge_iou_thr
        self.merge_ioa_thr = merge_ioa_thr
        self.inference_scale = inference_scale

        #self.engine.input_scale = inference_scale
        #self.engine.label_divisor = label_divisor
        #self.engine.ks = median_kernel_size
        #self.engine.mid_idx = (median_kernel_size - 1) // 2
        #self.engine.nms_threshold = nms_threshold
        #self.engine.nms_kernel = nms_kernel
        #self.engine.confidence_thr = confidence_thr

        # reset median queue for good measure
        #self.engine.reset()
        
    def create_matchers(self):
        matchers = []
        for thing_class in self.config['engine_params']['thing_list']:
            matchers.append(
                SequentialMatcher(thing_class, **self.config['matcher_params'])  
            )

        return matchers

    def infer_on_axis(self, volume, axis_name):
        mp.spawn(main_worker, nprocs=self.config['world_size'], args=(self.config, axis_name, volume, self.data))
        
        # grab the zarr stack that was filled in
        stack = self.data[f'panoptic_{axis_name}']
        
        # run backward matching and tracking
        axis = self.config['axes'][axis_name]
        matchers = self.create_matchers()
        
        trackers = [
            InstanceTracker(label, self.config['matcher_params']['label_divisor'], volume.shape, axis_name)
            for label in self.labels
        ]

        for matcher in matchers:
            matcher.assign_new = False
            matcher.force_connected = False
            
        zarr_queue = mp.Queue()
        zarr_proc = mp.Process(target=run_zarr_put, args=(zarr_queue, stack, axis))
        zarr_proc.start()
        
        #tracker_queue = mp.Queue()
        #tracker_proc = mp.Process(target=run_tracker, args=(tracker_queue, trackers))
        #tracker_proc.start()

        rev_indices = np.arange(0, stack.shape[axis])[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            pan_seg = zarr_take3d(stack, rev_idx, axis)    
            
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)
                    
            # don't overwrite last slice in the stack alone
            if rev_idx < (stack.shape[axis] - 1):
                zarr_queue.put((rev_idx, pan_seg))

            # track each instance for each class
            for tracker in trackers:
                tracker.update(pan_seg, rev_idx)
            #tracker_queue.put((rev_idx, pan_seg))
                
        zarr_queue.put((None, None))
        zarr_proc.join()
        
        #tracker_queue.put((None, None))
        #tracker_proc.join()

        return stack, trackers
        
def run_tracker(queue, trackers):
    while True:
        index, pan_seg = queue.get()
        if index is None:
            break
        else:
            for tracker in trackers:
                tracker.update(pan_seg, index)
                
    # finish tracking
    for tracker in trackers:
        tracker.finish()
        
def run_zarr_put(queue, stack, axis):
    while True:
        index, pan_seg = queue.get()
        if index is None:
            break
        else:
            zarr_put3d(stack, index, pan_seg, axis)
        
