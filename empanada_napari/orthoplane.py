import os
import zarr
import yaml
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada.data.utils import resize_by_factor

from empanada.inference import filters
from empanada.inference.engines import MultiScaleInferenceEngine
from empanada.inference.tracker import InstanceTracker
from empanada.inference.matcher import RLEMatcher
from empanada.array_utils import put
from empanada.inference.rle import pan_seg_to_rle_seg, rle_seg_to_pan_seg
from empanada.zarr_utils import *
from empanada.consensus import merge_objects3d

from napari.qt.threading import thread_worker

def instance_relabel(tracker):
    instance_id = 1
    instances = {}
    for instance_attr in tracker.instances.values():
        # vote on indices that should belong to an object
        runs_cat = np.stack([
            instance_attr['starts'], instance_attr['runs']
        ], axis=1)

        sort_idx = np.argsort(runs_cat[:, 0], kind='stable')
        runs_cat = runs_cat[sort_idx]

        # TODO: technically this could break the zarr_fill_instances function
        # if an object has a pixel in the bottom right corner of the Nth z slice
        # and a pixel in the top left corner of the N+1th z slice
        instances[instance_id] = {}
        instances[instance_id]['box'] = instance_attr['box']
        instances[instance_id]['starts'] = runs_cat[:, 0]
        instances[instance_id]['runs'] = runs_cat[:, 1]
        instance_id += 1

    return instances

@thread_worker
def stack_postprocessing(
    trackers,
    store_url,
    model_config,
    label_divisor=1000,
    min_size=200,
    min_extent=4
):
    labels = model_config['labels']
    class_names = model_config['class_names']
    if store_url is not None:
        data = zarr.open(store_url)
    else:
        data = None

    # create the final instance segmentations
    for class_id, class_name in zip(labels, class_names):
        # get the relevant trackers for the class_label
        print(f'Creating stack segmentation for class {class_name}...')

        class_tracker = None
        for axis_name, axis_trackers in trackers.items():
            for tracker in axis_trackers:
                if tracker.class_id == class_id:
                    class_tracker = tracker
                    break

        shape3d = class_tracker.shape3d

        # merge instances from orthoplane inference
        stack_tracker = InstanceTracker(class_id, label_divisor, shape3d, 'xy')
        stack_tracker.instances = instance_relabel(class_tracker)

        # inplace apply filters to final merged segmentation
        filters.remove_small_objects(stack_tracker, min_size=min_size)
        filters.remove_pancakes(stack_tracker, min_span=min_extent)

        print('N objects', len(stack_tracker.instances.keys()))

        # decode and fill the instances
        if data is not None:
            stack_vol = data.create_dataset(
                f'{class_name}_pred', shape=shape3d, dtype=np.uint64,
                overwrite=True, chunks=(1, None, None)
            )
        else:
            stack_vol = np.zeros(shape3d, dtype=np.uint64)

        zarr_fill_instances(stack_vol, stack_tracker.instances)
        yield stack_vol, class_name

@thread_worker
def tracker_consensus(
    trackers,
    store_url,
    model_config,
    label_divisor=1000,
    min_size=200,
    min_extent=4
):
    labels = model_config['labels']
    class_names = model_config['class_names']
    if store_url is not None:
        data = zarr.open(store_url)
    else:
        data = None

    # create the final instance segmentations
    for class_id, class_name in zip(labels, class_names):
        # get the relevant trackers for the class_label
        print(f'Creating consensus segmentation for class {class_name}...')

        class_trackers = []
        for axis_name, axis_trackers in trackers.items():
            for tracker in axis_trackers:
                if tracker.class_id == class_id:
                    class_trackers.append(tracker)

        shape3d = class_trackers[0].shape3d

        # merge instances from orthoplane inference
        consensus_tracker = InstanceTracker(class_id, label_divisor, shape3d, 'xy')
        consensus_tracker.instances = merge_objects3d(class_trackers)

        # inplace apply filters to final merged segmentation
        filters.remove_small_objects(consensus_tracker, min_size=min_size)
        filters.remove_pancakes(consensus_tracker, min_span=min_extent)

        print('N objects', len(consensus_tracker.instances.keys()))

        # decode and fill the instances
        if data is not None:
            consensus_vol = data.create_dataset(
                f'{class_name}_pred', shape=shape3d, dtype=np.uint64,
                overwrite=True, chunks=(1, None, None)
            )
        else:
            consensus_vol = np.zeros(shape3d, dtype=np.uint64)

        zarr_fill_instances(consensus_vol, consensus_tracker.instances)
        yield consensus_vol, class_name

class TestEngine:
    def __init__(
        self,
        model_config,
        inference_scale=1,
        label_divisor=1000,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.3,
        semantic_only=False,
        fine_boundaries=False,
        use_gpu=True
    ):
        # check whether GPU is available
        if torch.cuda.is_available() and use_gpu:
            device = 'gpu'
        else:
            device = 'cpu'

        # load the base and render models from file or url
        if os.path.isfile(model_config[f'base_model_{device}']):
            base_model = torch.jit.load(model_config[f'base_model_{device}'])
        else:
            base_model = torch.hub.load_state_dict_from_url(model_config[f'base_model_{device}'])

        if os.path.isfile(model_config[f'render_model_{device}']):
            render_model = torch.jit.load(model_config[f'render_model_{device}'])
        else:
            render_model = torch.hub.load_state_dict_from_url(model_config[f'render_model_{device}'])

        self.thing_list = model_config['thing_list']
        self.labels = model_config['labels']
        self.class_names = model_config['class_names']
        self.label_divisor = label_divisor
        self.padding_factor = model_config['padding_factor']
        self.inference_scale = inference_scale
        self.fine_boundaries = fine_boundaries

        if semantic_only:
            thing_list = []
        else:
            thing_list = self.thing_list

        # create the inference engine
        self.engine = MultiScaleInferenceEngine(
            base_model, render_model,
            thing_list=thing_list,
            median_kernel_size=1,
            label_divisor=label_divisor,
            nms_threshold=nms_threshold,
            nms_kernel=nms_kernel,
            confidence_thr=confidence_thr,
            padding_factor=self.padding_factor,
            coarse_boundaries=not fine_boundaries,
            device=device
        )

        # set the image transforms
        norms = model_config['norms']
        gray_channels = 1
        self.tfs = A.Compose([
            A.Normalize(**norms),
            ToTensorV2()
        ])

    def update_params(
        self,
        inference_scale,
        label_divisor,
        nms_threshold,
        nms_kernel,
        confidence_thr,
        fine_boundaries,
        semantic_only=False
    ):
        # note that input_scale is the variable name in the engine
        self.inference_scale = inference_scale
        self.engine.input_scale = inference_scale

        self.label_divisor = label_divisor
        self.engine.label_divisor = label_divisor

        self.nms_threshold = nms_threshold
        self.engine.nms_threshold = nms_threshold

        self.nms_kernel = nms_kernel
        self.engine.nms_kernel = nms_kernel

        self.confidence_thr = confidence_thr
        self.engine.confidence_thr = confidence_thr

        self.fine_boundaries = fine_boundaries
        self.engine.coarse_boundaries = not fine_boundaries

        if semantic_only:
            self.engine.thing_list = []
        else:
            self.engine.thing_list = self.thing_list

    def infer(self, image):
        # resize image to correct scale
        image = resize_by_factor(image, self.inference_scale)
        image = self.tfs(image=image)['image'].unsqueeze(0)

        # engine handles upsampling and padding
        pan_seg = self.engine(image, upsampling=self.inference_scale)
        return pan_seg.squeeze().cpu().numpy()

def run_forward_matchers(stack, axis, matchers, queue):
    while True:
        fill_index, pan_seg = queue.get()
        if pan_seg is None:
            continue
        elif isinstance(pan_seg, str):
            break
        else:
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)

            zarr_put3d(stack, fill_index, pan_seg, axis)

class OrthoPlaneEngine:
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
        if torch.cuda.is_available() and use_gpu:
            device = 'gpu'
        else:
            device = 'cpu'

        # load the base and render models from file or url
        if os.path.isfile(model_config[f'base_model_{device}']):
            base_model = torch.jit.load(model_config[f'base_model_{device}'])
        else:
            base_model = torch.hub.load_state_dict_from_url(model_config[f'base_model_{device}'])

        if os.path.isfile(model_config[f'render_model_{device}']):
            render_model = torch.jit.load(model_config[f'render_model_{device}'])
        else:
            render_model = torch.hub.load_state_dict_from_url(model_config[f'render_model_{device}'])

        self.thing_list = model_config['thing_list']
        self.labels = model_config['labels']
        self.class_names = model_config['class_names']
        self.label_divisor = label_divisor
        self.padding_factor = model_config['padding_factor']
        self.inference_scale = inference_scale

        # create the inference engine
        self.engine = MultiScaleInferenceEngine(
            base_model, render_model, self.thing_list, [1],
            inference_scale, median_kernel_size, label_divisor,
            stuff_area, void_label, nms_threshold, nms_kernel,
            confidence_thr, device=device
        )

        # set the image transforms
        norms = model_config['norms']
        gray_channels = 1
        self.tfs = A.Compose([
            A.Normalize(**norms),
            ToTensorV2()
        ])

        self.axes = {'xy': 0, 'xz': 1, 'yz': 2}
        self.merge_iou_thr = merge_iou_thr
        self.merge_ioa_thr = merge_ioa_thr
        self.force_connected = force_connected

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

        self.engine.input_scale = inference_scale
        self.engine.label_divisor = label_divisor
        self.engine.ks = median_kernel_size
        self.engine.mid_idx = (median_kernel_size - 1) // 2
        self.engine.nms_threshold = nms_threshold
        self.engine.nms_kernel = nms_kernel
        self.engine.confidence_thr = confidence_thr

        # reset median queue for good measure
        self.engine.reset()

    def create_matchers(self):
        matchers = []
        for thing_class in self.thing_list:
            matchers.append(
                SequentialMatcher(
                    thing_class, self.label_divisor, self.merge_iou_thr,
                    self.merge_ioa_thr, force_connected=self.force_connected
                )
            )

        return matchers

    def create_panoptic_stack(self, axis_name, shape3d):
        # faster IO with chunking only along
        # the given axis, orthogonal viewing is slow though
        if self.data is not None:
            chunks = [None, None, None]
            chunks[self.axes[axis_name]] = 1
            chunks = tuple(chunks)
            stack = self.data.create_dataset(
                f'panoptic_{axis_name}', shape=shape3d,
                dtype=np.uint64, chunks=chunks, overwrite=True
            )

        else:
            # we'll use uint32 for in memory segs
            stack = np.zeros(shape3d, dtype=np.uint64)

        return stack

    def infer_on_axis(self, volume, axis_name):
        axis = self.axes[axis_name]
        # create the dataloader
        dataset = ArrayData(volume, self.inference_scale, axis, self.tfs)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=False,
            drop_last=False, num_workers=0
        )

        # create the trackers
        # create a separate tracker for
        # each prediction axis and each segmentation class
        trackers = [
            InstanceTracker(label, self.label_divisor, volume.shape, axis_name)
            for label in self.labels
        ]

        matchers = self.create_matchers()

        # output stack
        stack = self.create_panoptic_stack(axis_name, volume.shape)

        queue = mp.Queue()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(stack, axis, matchers, queue))
        matcher_proc.start()

        print(f'Predicting {axis_name}...')

        fill_index = 0
        imshape = list(volume.shape)
        del imshape[axis]
        h, w = imshape

        fill_index = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            image = factor_pad(image, self.padding_factor)

            pan_seg = self.engine(image)
            if pan_seg is None:
                # building the queue
                queue.put((fill_index, pan_seg))
                continue
            else:
                # only support single scale (i.e. scale 1)
                pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding and unit dimensions
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        final_segs = self.engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        print(f'Propagating labels backward...')

        # set the matchers to not assign new labels
        # and not split disconnected components
        for matcher in matchers:
            matcher.assign_new = False
            matcher.force_connected = False

        rev_indices = np.arange(0, stack.shape[axis])[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            pan_seg = zarr_take3d(stack, rev_idx, axis)

            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)

            # leave the last slice in the stack alone
            if rev_idx < (stack.shape[axis] - 1):
                stack = zarr_put3d(stack, rev_idx, pan_seg, axis)

            # track each instance for each class
            for tracker in trackers:
                tracker.update(pan_seg, rev_idx)

        # finish tracking
        for tracker in trackers:
            tracker.finish()

        self.engine.reset()

        return stack, trackers
