import os
import zarr
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada.inference import filters
from empanada.inference.engines import PanopticDeepLabRenderEngine3d
from empanada.inference.tracker import InstanceTracker
from empanada.inference.postprocess import factor_pad
from empanada.inference.patterns import *

from napari.qt.threading import thread_worker

from empanada_napari.utils import Preprocessor

MODEL_DIR = os.path.join(os.path.expanduser('~'), '.empanada/configs')
torch.hub.set_dir(MODEL_DIR)

def main_worker(gpu, volume, axis_name, rle_stack, rle_out, config):
    config['gpu'] = gpu
    rank = gpu
    axis = config['axes'][axis_name]

    dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001',
                            world_size=config['world_size'], rank=rank)

    # load models and set engine class from file or url
    if os.path.isfile(config[f'base_model_gpu']):
        base_model = torch.jit.load(config[f'base_model_gpu'])
    else:
        base_model = torch.hub.load_state_dict_from_url(config[f'base_model_gpu'])

    if os.path.isfile(config[f'render_model_gpu']):
        render_model = torch.jit.load(config[f'render_model_gpu'])
    else:
        render_model = torch.hub.load_state_dict_from_url(config[f'render_model_gpu'])

    engine_cls = PanopticDeepLabRenderEngine3d
    torch.cuda.set_device(config['gpu'])

    preprocessor = Preprocessor(**config['norms'])

    # create the dataloader
    shape = volume.shape
    upsampling = config['inference_scale']
    dataset = VolumeDataset(volume, axis, preprocessor, scale=upsampling)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True,
        drop_last=False, num_workers=0, sampler=sampler
    )

    # if in main process, create matchers and process
    if rank == 0:
        thing_list = config['engine_params']['thing_list']
        matchers = create_matchers(thing_list, **config['matcher_params'])

        queue = mp.Queue()
        matcher_out, matcher_in = mp.Pipe()
        matcher_args = (
            matchers, queue, rle_stack, matcher_in,
            config['engine_params']['confidence_thr'],
            config['engine_params']['median_kernel_size'],
            config['engine_params']['labels'],
            config['engine_params']['label_divisor'],
            thing_list
        )
        matcher_proc = mp.Process(
            target=forward_multigpu,
            args=matcher_args
        )
        matcher_proc.start()

    # create the inference engine
    base_model = base_model.to(f'cuda:{gpu}')
    render_model = render_model.to(f'cuda:{gpu}')
    render_models = {'sem_logits': render_model}
    inference_engine = engine_cls(base_model, render_models, **config['engine_params'])

    n = 0
    iterator = dataloader if rank != 0 else tqdm(dataloader, total=len(dataloader))
    step = dist.get_world_size()
    total_len = shape[axis]
    for batch in iterator:
        image = batch['image'].to(f'cuda:{gpu}', non_blocking=True)
        h, w = batch['size']
        image = factor_pad(image, config['padding_factor'])

        output = inference_engine.infer(image)
        instance_cells = inference_engine.get_instance_cells(
            output['ctr_hmp'], output['offsets'], upsampling
        )

        # correctly resize the sem and instance_cells
        coarse_sem_logits = output['sem_logits']
        sem_logits = coarse_sem_logits.clone()
        features = output['semantic_x']
        sem, _ = inference_engine.upsample_logits(
            sem_logits, coarse_sem_logits,
            features, upsampling * 4
        )

        # get median semantic seg
        sems = all_gather(sem)
        instance_cells = all_gather(instance_cells)

        # drop last segs if unnecessary
        n += len(sems)
        stop = min(step, (total_len - n) + step)

        # clip off extras
        sems = sems[:stop]
        instance_cells = instance_cells[:stop]

        if rank == 0:
            # run the matching process
            for sem, cells in zip(sems, instance_cells):
                queue.put(
                    (sem.cpu()[..., :h, :w], cells.cpu()[..., :h, :w])
                )

        del sems, instance_cells

    # pass None to queue to mark the end of inference
    if rank == 0:
        queue.put(('finish', 'finish'))
        rle_stack = matcher_out.recv()[0]
        matcher_proc.join()

        print('Finished matcher')

        # send the rle stack back to the main process
        rle_out.put([rle_stack])

class MultiGPUEngine3d:
    def __init__(
        self,
        model_config,
        inference_scale=1,
        label_divisor=1000,
        median_kernel_size=5,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.3,
        force_connected=True,
        min_size=500,
        min_extent=4,
        fine_boundaries=False,
        semantic_only=False,
        store_url=None,
        save_panoptic=False
    ):
        # check whether GPU is available
        if not torch.cuda.device_count() > 1:
            raise Exception(f'MultiGPU inference requires multiple GPUs! Run torch.cuda.device_count()')

        device = 'gpu'

        # load the base and render models
        self.labels = model_config['labels']
        self.config = model_config
        self.config['base_model_url'] = model_config[f'base_model_{device}']
        self.config['render_model_url'] = model_config[f'render_model_{device}']

        self.config['engine_params'] = {}
        if semantic_only:
            self.config['engine_params']['thing_list'] = []
        else:
            self.config['engine_params']['thing_list'] = self.config['thing_list']

        self.config['inference_scale'] = inference_scale
        self.config['engine_params']['labels'] = self.labels
        self.config['engine_params']['label_divisor'] = label_divisor
        self.config['engine_params']['median_kernel_size'] = median_kernel_size
        self.config['engine_params']['stuff_area'] = stuff_area
        self.config['engine_params']['void_label'] = void_label
        self.config['engine_params']['nms_threshold'] = nms_threshold
        self.config['engine_params']['nms_kernel'] = nms_kernel
        self.config['engine_params']['confidence_thr'] = confidence_thr
        self.config['engine_params']['coarse_boundaries'] = not fine_boundaries

        self.axes = {'xy': 0, 'xz': 1, 'yz': 2}
        self.config['axes'] = self.axes
        self.config['matcher_params'] = {}

        self.config['matcher_params']['label_divisor'] = label_divisor
        self.config['matcher_params']['merge_iou_thr'] = 0.25
        self.config['matcher_params']['merge_ioa_thr'] = 0.25
        self.config['force_connected'] = force_connected

        self.config['world_size'] = torch.cuda.device_count()

        self.min_size = min_size
        self.min_extent = min_extent

        self.save_panoptic = save_panoptic
        if store_url is not None:
            self.zarr_store = zarr.open(store_url, mode='w')
        else:
            self.zarr_store = None

        self.set_dtype()

    def set_dtype(self):
        # maximum possible value in panoptic seg
        max_index = self.config['matcher_params']['label_divisor'] * (1 + max(self.labels))
        if max_index < 2 ** 8:
            self.dtype = np.uint8
        elif max_index < 2 ** 16:
            self.dtype = np.uint16
        elif max_index < 2 ** 32:
            self.dtype = np.uint32
        else:
            self.dtype = np.uint64

    def create_trackers(self, shape3d, axis_name):
        label_divisor = self.config['engine_params']['label_divisor']
        trackers = [
            InstanceTracker(label, label_divisor, shape3d, axis_name)
            for label in self.labels
        ]
        return trackers

    def create_panoptic_stack(self, axis_name, shape3d):
        # faster IO with chunking only along
        # the given axis, orthogonal viewing is slow though
        if self.zarr_store is not None and self.save_panoptic:
            chunks = [None, None, None]
            chunks[self.axes[axis_name]] = 1
            stack = self.zarr_store.create_dataset(
                f'panoptic_{axis_name}', shape=shape3d,
                dtype=self.dtype, chunks=tuple(chunks), overwrite=True
            )
        elif self.save_panoptic:
            # we'll use uint32 for in memory segs
            stack = np.zeros(shape3d, dtype=self.dtype)
        else:
            stack = None

        return stack

    def infer_on_axis(self, volume, axis_name):
        ctx = mp.get_context('spawn')

        # create a pipe to get rle stack from main GPU process
        rle_out = ctx.Queue()

        # launch the GPU processes
        rle_stack = []
        context = mp.spawn(
            main_worker, nprocs=self.config['world_size'],
            args=(volume, axis_name, rle_stack, rle_out, self.config),
            join=False
        )

        # grab the zarr stack that was filled in
        rle_stack = rle_out.get()[0]
        context.join()

        # run backward matching and tracking
        print('Propagating labels backward...')
        axis = self.config['axes'][axis_name]
        matchers = create_matchers(self.config['thing_list'], **self.config['matcher_params'])
        trackers = self.create_trackers(volume.shape, axis_name)
        stack = self.create_panoptic_stack(axis_name, volume.shape)

        axis_len = volume.shape[axis]
        for index,rle_seg in tqdm(backward_matching(rle_stack, matchers, axis_len), total=axis_len):
            update_trackers(rle_seg, index, trackers, axis, stack)

        finish_tracking(trackers)
        for tracker in trackers:
            filters.remove_small_objects(tracker, min_size=self.min_size)
            filters.remove_pancakes(tracker, min_span=self.min_extent)

        return stack, trackers
