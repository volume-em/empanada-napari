import os, platform
import zarr
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
from empanada.inference import rle
from empanada.inference.tile import Tiler
from empanada.inference.patterns import *
from empanada.consensus import merge_objects_from_tiles, merge_semantic_from_tiles

from napari.qt.threading import thread_worker
from empanada_napari.utils import Preprocessor, load_model_to_device

MODEL_DIR = os.path.join(os.path.expanduser('~'), '.empanada')
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
        # only applies to yz axis
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
    chunk_size=(256, 256, 256)
):
    r"""Relabels and filters each class defined in trackers. Yields a numpy
    or zarr volume along with the name of the class that is segmented.
    """
    thing_list = model_config['thing_list']
    class_names = model_config['class_names']
    if store_url is not None:
        zarr_store = zarr.open(store_url)
    else:
        zarr_store = None

    # create the final instance segmentations
    for class_id, class_name in class_names.items():
        print(f'Creating stack segmentation for class {class_name}...')

        class_tracker = get_axis_trackers_by_class(trackers, class_id)[0]
        shape3d = class_tracker.shape3d

        # merge instances from orthoplane inference
        stack_tracker = InstanceTracker(class_id, label_divisor, shape3d, 'xy')
        stack_tracker.instances = instance_relabel(class_tracker)

        # inplace apply filters to final merged segmentation
        if class_id in thing_list:
            filters.remove_small_objects(stack_tracker, min_size=min_size)
            filters.remove_pancakes(stack_tracker, min_span=min_extent)
            class_dtype = dtype
        else:
            class_dtype = np.uint8 

        print(f'Total {class_name} objects {len(stack_tracker.instances.keys())}')

        # decode and fill the instances
        if zarr_store is not None:
            stack_vol = zarr_store.create_dataset(
                f'{class_name}', shape=shape3d, dtype=class_dtype,
                overwrite=True, chunks=chunk_size
            )
        else:
            stack_vol = np.zeros(shape3d, dtype=dtype)

        fill_volume(stack_vol, stack_tracker.instances)

        yield stack_vol, class_name, stack_tracker.instances

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
    chunk_size=(256, 256, 256)
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
    for class_id, class_name in class_names.items():
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
            class_dtype = dtype
        else:
            consensus_tracker = create_semantic_consensus(class_trackers, pixel_vote_thr)
            class_dtype = np.uint8

        print(f'Total {class_name} objects {len(consensus_tracker.instances.keys())}')

        # decode and fill the instances
        if zarr_store is not None:
            consensus_vol = zarr_store.create_dataset(
                f'{class_name}', shape=shape3d, dtype=class_dtype,
                overwrite=True, chunks=chunk_size
            )
        else:
            consensus_vol = np.zeros(shape3d, dtype=dtype)

        fill_volume(consensus_vol, consensus_tracker.instances)

        yield consensus_vol, class_name, consensus_tracker.instances

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
        tile_size=0,
        use_gpu=True,
        use_quantized=False
    ):
        # check whether GPU is available
        device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
        if use_quantized and str(device) == 'cpu' and model_config.get('model_quantized') is not None:
            model_url = model_config['model_quantized']
        else:
            model_url = model_config['model']

        model = load_model_to_device(model_url, device)
        model = model.to(device)

        self.thing_list = model_config['thing_list']
        self.labels = model_config['labels']
        self.class_names = model_config['class_names']
        self.label_divisor = label_divisor
        self.padding_factor = model_config['padding_factor']
        self.inference_scale = inference_scale
        self.fine_boundaries = fine_boundaries
        self.tile_size = tile_size

        if semantic_only:
            thing_list = []
        else:
            thing_list = self.thing_list

        # create the inference engine
        self.engine = PanopticDeepLabRenderEngine(
            model, thing_list=thing_list,
            label_divisor=label_divisor,
            nms_threshold=nms_threshold,
            nms_kernel=nms_kernel,
            confidence_thr=confidence_thr,
            padding_factor=self.padding_factor,
            coarse_boundaries=not fine_boundaries
        )

        # set the image transforms
        norms = model_config['norms']
        self.preprocessor = Preprocessor(**norms)

    def update_params(
        self,
        inference_scale,
        label_divisor,
        nms_threshold,
        nms_kernel,
        confidence_thr,
        fine_boundaries,
        semantic_only=False,
        tile_size=0
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

        self.tile_size = tile_size

    def force_connected(self, pan_seg):
        for label in self.engine.thing_list:
            # convert from pan_seg to instance_seg
            min_id = label * self.label_divisor
            max_id = min_id + self.label_divisor

            # zero all objects/semantic segs outside of instance_id range
            instance_seg = pan_seg.copy()
            outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
            instance_seg[outside_mask] = 0

            # relabel connected components
            instance_seg = rle.connected_components(instance_seg).astype(np.int32)
            instance_seg[instance_seg > 0] += min_id
            pan_seg[instance_seg > 0] = instance_seg[instance_seg > 0]

        return pan_seg

    def infer(self, image):
        # engine handles upsampling and padding
        if self.tile_size > 0 and any([s > self.tile_size for s in image.shape]):
            print('Tiling image for inference...')
            tiler = Tiler(
                image.shape, tile_size=self.tile_size, 
                overlap_width=min(128, int(self.tile_size * 0.1))
            )

            rle_segs = []
            for i in tqdm(range(len(tiler))):
                tile = tiler(image, i)
                tile_size = tile.shape
                tile = resize_by_factor(tile, self.inference_scale)
                tile = self.preprocessor(tile)['image'].unsqueeze(0)

                tile_pan_seg = self.engine(tile, tile_size, upsampling=self.inference_scale)
                tile_pan_seg = tile_pan_seg.squeeze().cpu().numpy().astype(np.int32)
                tile_rle_seg = rle.pan_seg_to_rle_seg(
                    tile_pan_seg, self.labels, self.label_divisor, self.engine.thing_list
                )
                tile_rle_seg = tiler.translate_rle_seg(tile_rle_seg, i)
                rle_segs.append(tile_rle_seg)

            # merge the tiles with consensus
            rle_seg = {}
            for label in self.labels:
                if label in self.engine.thing_list:
                    rle_seg[label] = merge_objects_from_tiles(
                        [rs[label] for rs in rle_segs], tiler.overlap_rle
                    )
                else:
                    rle_seg[label] = merge_semantic_from_tiles(
                        [rs[label] for rs in rle_segs]
                    )

            pan_seg = rle.rle_seg_to_pan_seg(rle_seg, image.shape)
            return pan_seg
        else:
            size = image.shape
            # resize image to correct scale
            image = resize_by_factor(image, self.inference_scale)
            image = self.preprocessor(image)['image'].unsqueeze(0)
            pan_seg = self.engine(image, size, upsampling=self.inference_scale)
            return self.force_connected(pan_seg.squeeze().cpu().numpy().astype(np.int32))

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
        use_quantized=False,
        store_url=None,
        chunk_size=(256, 256, 256),
        save_panoptic=False
    ):
        # check whether GPU is available
        device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
        if use_quantized and str(device) == 'cpu' and model_config.get('model_quantized') is not None:
            model_url = model_config['model_quantized']
        else:
            model_url = model_config['model']

        model = load_model_to_device(model_url, device)
        model = model.to(device)

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
        self.engine = PanopticDeepLabRenderEngine3d(
            model, thing_list=self.thing_list,
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
        self.chunk_size = chunk_size
        if store_url is not None:
            self.zarr_store = zarr.open(store_url, mode='w')
        else:
            self.zarr_store = None

        self.dtype = np.int32

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
        chunk_size,
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
        self.chunk_size = chunk_size
        if store_url is not None:
            self.zarr_store = zarr.open(store_url, mode='w')
        else:
            self.zarr_store = None

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
            stack = self.zarr_store.create_dataset(
                f'panoptic_{axis_name}', shape=shape3d,
                dtype=self.dtype, chunks=self.chunk_size, overwrite=True
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

        final_segs = self.engine.end(self.inference_scale)
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
            update_trackers(rle_seg, index, trackers)

        finish_tracking(trackers)
        for tracker in trackers:
            filters.remove_small_objects(tracker, min_size=self.min_size)
            filters.remove_pancakes(tracker, min_span=self.min_extent)

        if stack is not None:
            print('Writing panoptic segmentation.')
            fill_panoptic_volume(stack, trackers)

        self.engine.reset()

        return stack, trackers