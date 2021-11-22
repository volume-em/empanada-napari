import os
import zarr
import yaml
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

from empanada_napari.dask_utils import *

from tqdm import tqdm

def tracker_consensus(trackers, store_url, model_config, label_divisor=1000):
    labels = model_config['labels']
    class_names = model_config['class_names']
    data = zarr.open(store_url)

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
        
        # merge instances from orthoplane inference if applicable
        if len(class_trackers) > 1:
            consensus_tracker = InstanceTracker(class_id, label_divisor, shape3d, 'xy')
            
            consensus_tracker.instances = merge_objects3d(class_trackers)

            # apply filters to final merged segmentation
            #if filter_names:
            #    for filt,kwargs in zip(filter_names, filter_kwargs):
            #        filters.__dict__[filt](consensus_tracker, **kwargs)
        else:
            consensus_tracker = class_trackers[0]
            

        # decode and fill the instances
        consensus_vol = data.create_dataset(
            f'{class_name}_pred', shape=shape3d, dtype=np.uint64,
            overwrite=True, chunks=(1, None, None)
        )
        zarr_fill_instances(consensus_vol, consensus_tracker.instances)

    return consensus_vol

class TestEngine:
    def __init__(
        self,
        model_config,
        inference_scale=1,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=3,
        confidence_thr=0.3,
    ):
        # load the base and render models
        base_model = torch.hub.load_state_dict_from_url(model_config['base_model'])
        render_model = torch.hub.load_state_dict_from_url(model_config['render_model'])
        thing_list = model_config['thing_list']
        self.thing_list = thing_list
        self.labels = model_config['labels']
        self.class_names = model_config['class_names']
        self.label_divisor = label_divisor
        self.padding_factor = model_config['padding_factor']

        # create the inference engine
        self.engine = MultiScaleInferenceEngine(
            base_model, render_model, thing_list, [1],
            inference_scale, 1, label_divisor,
            stuff_area, void_label, nms_threshold, nms_kernel,
            confidence_thr, device='cpu'
        )

        # set the image transforms
        norms = model_config['norms']
        gray_channels = 1
        self.tfs = A.Compose([
            A.Normalize(**norms),
            ToTensorV2()
        ])

    def infer(self, image):
        image = self.tfs(image=image)['image'].unsqueeze(0)
        h, w = image.size()[2:]
        image = factor_pad(image, self.padding_factor)
            
        pan_seg = self.engine(image)

        # only support single scale (i.e. scale 1)
        pan_seg = pan_seg[0]
        pan_seg = pan_seg.squeeze()[:h, :w] # remove padding and unit dimensions
        pan_seg = pan_seg.cpu().numpy() # move to cpu if not already

        return pan_seg

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
        filters=None
    ):
        if os.path.isdir(store_url):
            self.data = zarr.open(store_url, mode='r+')
        else:
            self.data = zarr.open(store_url, mode='w')

        # load the base and render models
        base_model = torch.hub.load_state_dict_from_url(model_config['base_model'])
        render_model = torch.hub.load_state_dict_from_url(model_config['render_model'])
        thing_list = model_config['thing_list']
        self.thing_list = thing_list
        self.labels = model_config['labels']
        self.class_names = model_config['class_names']
        self.label_divisor = label_divisor
        self.padding_factor = model_config['padding_factor']

        # create the inference engine
        self.engine = MultiScaleInferenceEngine(
            base_model, render_model, thing_list, [1],
            inference_scale, median_kernel_size, label_divisor,
            stuff_area, void_label, nms_threshold, nms_kernel,
            confidence_thr, device='cpu'
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
        chunks = [None, None, None]
        chunks[self.axes[axis_name]] = 1
        chunks = tuple(chunks)
        stack = self.data.create_dataset(
            f'panoptic_{axis_name}', shape=shape3d, 
            dtype=np.uint64, chunks=chunks, overwrite=True
        )

        return stack

    def infer_on_axis(self, volume, axis_name):
        axis = self.axes[axis_name]
        # create the dataloader
        dataset = DaskData(volume, axis, self.tfs)
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

        print(f'Predicting {axis_name}...')

        fill_index = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[2:]
            image = factor_pad(image, self.padding_factor)
            
            pan_seg = self.engine(image)
            if pan_seg is None:
                # building the queue
                continue

            # only support single scale (i.e. scale 1)
            pan_seg = pan_seg[0]
            pan_seg = pan_seg.squeeze()[:h, :w] # remove padding and unit dimensions
            pan_seg = pan_seg.cpu().numpy() # move to cpu if not already
            
            # update the panoptic segmentations for each
            # thing class by passing it through matchers
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)
            
            # store the result
            stack = zarr_put3d(stack, fill_index, pan_seg, axis)
            
            # increment the fill_index
            fill_index += 1
            
        # if inference engine has a queue,
        # then there will be a few remaining
        # segmentations to fill in
        final_segs = self.engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w]
                pan_seg = pan_seg.cpu().numpy()
                
                for matcher in matchers:
                    pan_seg = matcher(pan_seg)
                    
                stack = zarr_put3d(stack, fill_index, pan_seg, axis)
                fill_index += 1
                
        print(f'Propagating labels backward...')

        # set the matchers to not assign new labels
        # and not split disconnected components
        for matcher in matchers:
            matcher.assign_new = False
            matcher.force_connected = False

        rev_indices = np.arange(0, stack.shape[axis] - 1)[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            pan_seg = zarr_take3d(stack, rev_idx, axis)
            
            for matcher in matchers:
                pan_seg = matcher(pan_seg)

            stack = zarr_put3d(stack, rev_idx, pan_seg, axis)
            
            # track each instance for each class
            for tracker in trackers:
                tracker.update(pan_seg, rev_idx)
            
        # finish tracking
        for tracker in trackers:
            tracker.finish()

        self.engine.reset()

        return stack, trackers

        

