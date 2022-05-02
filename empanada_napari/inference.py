import os, platform
import zarr
import yaml
import numpy as np
import torch

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada.data.utils import resize_by_factor

from empanada.inference import filters
from empanada.inference.engines import (
    PanopticDeepLabRenderEngine, PanopticDeepLabRenderEngine3d
)
from empanada.inference.tracker import InstanceTracker
from empanada.array_utils import put
from empanada.inference.patterns import *

from napari.qt.threading import thread_worker
from empanada_napari.utils import Preprocessor

MODEL_DIR = os.path.join(os.path.expanduser('~'), '.empanada/configs')
torch.hub.set_dir(MODEL_DIR)

def instance_relabel(tracker):
    r"""Relabels instances starting from 1"""
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
    min_extent=4,
    dtype=np.uint32,
    chunk_size=(96, 96, 96)
):
    r"""Relabels and filters each class defined in trackers. Yields a numpy
    or zarr volume along with the name of the class that is segmented.
    """
    labels = model_config['labels']
    class_names = model_config['class_names']
    if store_url is not None:
        zarr_store = zarr.open(store_url)
    else:
        zarr_store = None

    # create the final instance segmentations
    for class_id, class_name in zip(labels, class_names):
        print(f'Creating stack segmentation for class {class_name}...')

        class_tracker = get_axis_trackers_by_class(trackers, class_id)[0]
        shape3d = class_tracker.shape3d

        # merge instances from orthoplane inference
        stack_tracker = InstanceTracker(class_id, label_divisor, shape3d, 'xy')
        stack_tracker.instances = instance_relabel(class_tracker)

        # inplace apply filters to final merged segmentation
        filters.remove_small_objects(stack_tracker, min_size=min_size)
        filters.remove_pancakes(stack_tracker, min_span=min_extent)

        print(f'Total {class_name} objects {len(stack_tracker.instances.keys())}')

        # decode and fill the instances
        if zarr_store is not None:
            stack_vol = zarr_store.create_dataset(
                f'{class_name}_pred', shape=shape3d, dtype=dtype,
                overwrite=True, chunks=chunk_size
            )
        else:
            stack_vol = np.zeros(shape3d, dtype=dtype)

        fill_volume(stack_vol, stack_tracker.instances)

        yield stack_vol, class_name

@thread_worker
def tracker_consensus(
    trackers,
    store_url,
    model_config,
    label_divisor=1000,
    pixel_vote_thr=2,
    cluster_iou_thr=0.75,
    allow_one_view=False,
    min_size=200,
    min_extent=4,
    dtype=np.uint32,
    chunk_size=(96, 96, 96)
):
    r"""Calculate the orthoplane consensus from trackers. Yields a numpy
    or zarr volume along with the name of the class that is segmented.
    """
    labels = model_config['labels']
    thing_list = model_config['thing_list']
    class_names = model_config['class_names']
    if store_url is not None:
        zarr_store = zarr.open(store_url)
    else:
        zarr_store = None

    # create the final instance segmentations
    for class_id, class_name in zip(labels, class_names):
        # get the relevant trackers for the class_label
        print(f'Creating consensus segmentation for class {class_name}...')

        class_trackers = get_axis_trackers_by_class(trackers, class_id)
        shape3d = class_trackers[0].shape3d

        # consensus from orthoplane
        if class_id in thing_list:
            consensus_tracker = create_instance_consensus(
                class_trackers, pixel_vote_thr, cluster_iou_thr, allow_one_view
            )
            filters.remove_small_objects(consensus_tracker, min_size=min_size)
            filters.remove_pancakes(consensus_tracker, min_span=min_extent)
        else:
            consensus_tracker = create_semantic_consensus(class_trackers, args.pixel_vote_thr)

        print(f'Total {class_name} objects {len(consensus_tracker.instances.keys())}')

        # decode and fill the instances
        if zarr_store is not None:
            consensus_vol = zarr_store.create_dataset(
                f'{class_name}_pred', shape=shape3d, dtype=dtype,
                overwrite=True, chunks=chunk_size
            )
        else:
            consensus_vol = np.zeros(shape3d, dtype=dtype)

        fill_volume(consensus_vol, consensus_tracker.instances)

        yield consensus_vol, class_name

class Engine2d:
    r"""Engine for 2D and parameter testing."""
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
            print('Testing on CPU')
            device = 'cpu'

        # load the base and render models from file or url
        if os.path.isfile(model_config[f'base_model_{device}']):
            print(f'base_model_{device}', model_config[f'base_model_{device}'])
            base_model = torch.jit.load(model_config[f'base_model_{device}'])
        else:
            base_model = torch.hub.load_state_dict_from_url(model_config[f'base_model_{device}'])

        if os.path.isfile(model_config[f'render_model_{device}']):
            print(f'render_model_{device}', model_config[f'render_model_{device}'])
            render_model = torch.jit.load(model_config[f'render_model_{device}'])
        else:
            render_model = torch.hub.load_state_dict_from_url(model_config[f'render_model_{device}'])

        # move them to gpu if needed
        if device == 'gpu':
            base_model = base_model.cuda()
            render_model = render_model.cuda()

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
        render_models = {'sem_logits': render_model}
        self.engine = PanopticDeepLabRenderEngine(
            base_model, render_models,
            thing_list=thing_list,
            label_divisor=label_divisor,
            nms_threshold=nms_threshold,
            nms_kernel=nms_kernel,
            confidence_thr=confidence_thr,
            padding_factor=self.padding_factor,
            coarse_boundaries=not fine_boundaries
        )

        # set the image transforms
        norms = model_config['norms']
        gray_channels = 1
        self.preprocessor = Preprocessor(**norms)

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
        size = image.shape
        image = resize_by_factor(image, self.inference_scale)
        image = self.preprocessor(image)['image'].unsqueeze(0)

        # engine handles upsampling and padding
        pan_seg = self.engine(image, size, upsampling=self.inference_scale)
        return pan_seg.squeeze().cpu().numpy()

class Engine3d:
    r"""Engine for 3D ortho-plane and stack inference"""
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
        use_gpu=True,
        store_url=None,
        save_panoptic=False
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

        # move them to gpu if needed
        if device == 'gpu':
            base_model = base_model.cuda()
            render_model = render_model.cuda()

        self.model_config = model_config
        self.labels = model_config['labels']
        self.class_names = model_config['class_names']
        self.label_divisor = label_divisor
        self.padding_factor = model_config['padding_factor']
        self.inference_scale = inference_scale

        # downgrade all thing classes
        if semantic_only:
            self.thing_list = []
        else:
            self.thing_list = model_config['thing_list']

        # create the inference engine
        render_models = {'sem_logits': render_model}
        self.engine = PanopticDeepLabRenderEngine3d(
            base_model, render_models,
            thing_list=self.thing_list,
            median_kernel_size=median_kernel_size,
            label_divisor=label_divisor,
            nms_threshold=nms_threshold,
            nms_kernel=nms_kernel,
            confidence_thr=confidence_thr,
            padding_factor=self.padding_factor,
            coarse_boundaries=not fine_boundaries
        )

        # set the image transforms
        norms = model_config['norms']
        gray_channels = 1
        self.preprocessor = Preprocessor(**norms)

        self.axes = {'xy': 0, 'xz': 1, 'yz': 2}
        self.merge_iou_thr = 0.25
        self.merge_ioa_thr = 0.25
        self.force_connected = force_connected
        self.min_size = min_size
        self.min_extent = min_extent
        self.fine_boundaries = fine_boundaries

        self.save_panoptic = save_panoptic
        if store_url is not None:
            self.zarr_store = zarr.open(store_url, mode='w')
        else:
            self.zarr_store = None

        self.set_dtype()

    def set_dtype(self):
        # maximum possible value in panoptic seg
        max_index = self.label_divisor * (1 + max(self.labels))
        if max_index < 2 ** 8:
            self.dtype = np.uint8
        elif max_index < 2 ** 16:
            self.dtype = np.uint16
        elif max_index < 2 ** 32:
            self.dtype = np.uint32
        else:
            self.dtype = np.uint64

    def update_params(
        self,
        inference_scale,
        label_divisor,
        median_kernel_size,
        nms_threshold,
        nms_kernel,
        confidence_thr,
        min_size,
        min_extent,
        fine_boundaries,
        semantic_only,
        store_url,
        save_panoptic
    ):
        self.label_divisor = label_divisor
        self.inference_scale = inference_scale
        self.min_size = min_size
        self.min_extent = min_extent
        self.fine_boundaries = fine_boundaries

        self.engine.label_divisor = label_divisor
        self.engine.ks = median_kernel_size
        self.engine.mid_idx = (median_kernel_size - 1) // 2
        self.engine.nms_threshold = nms_threshold
        self.engine.nms_kernel = nms_kernel
        self.engine.confidence_thr = confidence_thr
        self.engine.coarse_boundaries = not fine_boundaries

        if semantic_only:
            self.thing_list = []
            self.engine.thing_list = []
        else:
            self.thing_list = self.model_config['thing_list']
            self.engine.thing_list = self.thing_list

        # reset median queue for good measure
        self.engine.reset()

        self.save_panoptic = save_panoptic
        if store_url is not None:
            self.zarr_store = zarr.open(store_url, mode='w')
        else:
            self.zarr_store = None

        self.set_dtype()

    def create_trackers(self, shape3d, axis_name):
        trackers = [
            InstanceTracker(label, self.label_divisor, shape3d, axis_name)
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
        axis = self.axes[axis_name]
        # create the dataloader
        dataset = VolumeDataset(volume, axis, self.preprocessor, scale=self.inference_scale)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=False,
            drop_last=False, num_workers=0
        )

        # create necessary matchers and trackers
        trackers = self.create_trackers(volume.shape, axis_name)
        matchers = create_matchers(
            self.thing_list, self.label_divisor,
            self.merge_iou_thr, self.merge_ioa_thr
        )
        stack = self.create_panoptic_stack(axis_name, volume.shape)

        if platform.system() == "Darwin":
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass

        # setup matcher for multiprocessing
        queue = mp.Queue()
        rle_stack = []
        matcher_out, matcher_in = mp.Pipe()
        matcher_args = (
            matchers, queue, rle_stack, matcher_in,
            self.labels, self.label_divisor, self.thing_list
        )
        matcher_proc = mp.Process(target=forward_matching, args=matcher_args)
        matcher_proc.start()

        print(f'Predicting {axis_name}...')
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            size = batch['size']
            pan_seg = self.engine(image, size, self.inference_scale)

            if pan_seg is None:
                # building the median queue
                queue.put(None)
                continue
            else:
                pan_seg = pan_seg.squeeze().cpu().numpy()
                queue.put(pan_seg)

        final_segs = self.engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg.squeeze().cpu().numpy()
                queue.put(pan_seg)

        # finish and close forward matching process
        queue.put('finish')
        rle_stack = matcher_out.recv()[0]
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')
        axis_len = volume.shape[axis]
        for index,rle_seg in tqdm(backward_matching(rle_stack, matchers, axis_len), total=axis_len):
            update_trackers(rle_seg, index, trackers, axis, stack)

        finish_tracking(trackers)
        for tracker in trackers:
            filters.remove_small_objects(tracker, min_size=self.min_size)
            filters.remove_pancakes(tracker, min_span=self.min_extent)

        self.engine.reset()

        return stack, trackers
