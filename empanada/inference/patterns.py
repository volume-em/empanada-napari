import numpy as np
import torch
import zarr
import torch.distributed as dist
from empanada.array_utils import put, numpy_fill_instances
from empanada.zarr_utils import zarr_fill_instances
from empanada.inference import filters
from empanada.inference.engines import _MedianQueue
from empanada.inference.matcher import RLEMatcher
from empanada.inference.tracker import InstanceTracker
from empanada.inference.postprocess import merge_semantic_and_instance
from empanada.inference.rle import pan_seg_to_rle_seg, rle_seg_to_pan_seg
from empanada.consensus import merge_objects_from_trackers, merge_semantic_from_trackers

__all__ = [
    'create_matchers',
    'create_axis_trackers',
    'apply_matchers',
    'forward_matching',
    'backward_matching',
    'update_trackers',
    'finish_tracking',
    'apply_filters',
    'get_axis_trackers_by_class',
    'create_instance_consensus',
    'create_semantic_consensus',
    'fill_volume',
    'fill_panoptic_volume',
    'all_gather',
    'forward_multigpu'
]

def create_matchers(thing_list, label_divisor, merge_iou_thr, merge_ioa_thr):
    r"""Create matchers for all instances classes."""
    matchers = [
        RLEMatcher(thing_class, label_divisor, merge_iou_thr, merge_ioa_thr)
        for thing_class in thing_list
    ]
    return matchers

def create_axis_trackers(axes, class_labels, label_divisor, shape):
    r"""Create a dictionary of trackers for all classes. Each
    key is an axis_name ('xy', 'xz', 'yz') and keys are a list
    of trackers for each class.
    """
    trackers = {}
    for axis_name, axis in axes.items():
        trackers[axis_name] = [
            InstanceTracker(class_id, label_divisor, shape, axis_name)
            for class_id in class_labels
        ]

    return trackers

def apply_matchers(rle_seg, matchers):
    r"""Matches instances in the given segmentation to
    instances from the previous segmentation.
    """
    for matcher in matchers:
        class_id = matcher.class_id
        if matcher.target_rle is None:
            matcher.initialize_target(rle_seg[class_id])
        else:
            rle_seg[class_id] = matcher(rle_seg[class_id])

    return rle_seg

def forward_matching(
    matchers,
    queue,
    rle_stack,
    matcher_in,
    labels,
    label_divisor,
    thing_list
):
    r"""Uses multiprocessing.Queue to receive predicted segmentations,
    convert them to run length encodings, and match instances across
    planes. Runs in parallel with model inference.
    """
    # go until queue gets the kill signal
    while True:
        # create the rle_seg
        pan_seg = queue.get()

        if pan_seg is None:
            # building the median filter queue
            continue
        elif type(pan_seg) == str:
            # all images have been matched!
            break
        else:
            rle_seg = pan_seg_to_rle_seg(
                pan_seg, labels, label_divisor, thing_list, force_connected=True
            )
            rle_seg = apply_matchers(rle_seg, matchers)
            rle_stack.append(rle_seg)

    matcher_in.send([rle_stack])
    matcher_in.close()

def backward_matching(
    rle_stack,
    matchers,
    axis_len
):
    r"""Generator function that matches instances backward through the stack
    and yields each matched segmentation in turn.
    """
    # set the matchers to not assign new labels
    for matcher in matchers:
        matcher.target_rle = None
        matcher.assign_new = False

    rev_indices = np.arange(0, axis_len)[::-1]
    for rev_idx in rev_indices:
        rev_idx = rev_idx.item()
        rle_seg = rle_stack[rev_idx]
        rle_seg = apply_matchers(rle_seg, matchers)

        yield rev_idx, rle_seg

def update_trackers(
    rle_seg,
    index,
    trackers,
):
    r"""Updates trackers for a given axis with forward and
    backward matched instance segmentations. 
    """
    # track each instance for each class
    for tracker in trackers:
        class_id = tracker.class_id
        tracker.update(rle_seg[class_id], index)

def finish_tracking(trackers):
    r"""Ends tracking of instances."""
    for tracker in trackers:
        tracker.finish()

def apply_filters(
    tracker,
    filters_dict
):
    r"""Applies a list of filters to the given tracker."""
    if filters_dict is not None:
        for filt in filters_dict:
            name = filt['name']
            kwargs = {k: v for k,v in filt.items() if k != 'name'}

            # applied in-place
            filters.__dict__[name](tracker, **kwargs)

def get_axis_trackers_by_class(trackers, class_id):
    r"""Takes a dictionary of trackers (i.e., output of
    create_axis_trackers) and returns trackers across
    all axes that correspond to a particular segmentation
    class.
    """
    class_trackers = []
    for axis_name, axis_trackers in trackers.items():
        for tracker in axis_trackers:
            if tracker.class_id == class_id:
                class_trackers.append(tracker)

    return class_trackers

def create_instance_consensus(
    class_trackers,
    pixel_vote_thr=2,
    cluster_iou_thr=0.75,
    bypass=False
):
    r"""Applies the instance consensus algorithm to a list
    of trackers and returns a new tracker with the result.
    """
    class_id = class_trackers[0].class_id
    label_divisor = class_trackers[0].label_divisor
    shape = class_trackers[0].shape3d

    consensus_tracker = InstanceTracker(class_id, label_divisor, shape, 'xy')
    consensus_tracker.instances = merge_objects_from_trackers(
        class_trackers, pixel_vote_thr, cluster_iou_thr, bypass
    )

    return consensus_tracker

def create_semantic_consensus(
    class_trackers,
    pixel_vote_thr=2
):
    r"""Applies the semantic consensus algorithm (a simple vote) to a list
    of trackers and returns a new tracker with the result.
    """
    class_id = class_trackers[0].class_id
    label_divisor = class_trackers[0].label_divisor
    shape = class_trackers[0].shape3d

    consensus_tracker = InstanceTracker(class_id, label_divisor, shape, 'xy')
    consensus_tracker.instances = merge_semantic_from_trackers(class_trackers, pixel_vote_thr)

    return consensus_tracker

def fill_volume(volume, instances, processes=4):
    r"""Fills a numpy or zarr array with the given
    run length encoded instances. Runs in-place.
    """
    if isinstance(volume, np.ndarray):
        numpy_fill_instances(volume, instances)
    elif isinstance(volume, zarr.Array):
        zarr_fill_instances(volume, instances, processes)
    else:
        raise Exception(f'Unknown volume type of {type(volume)}')

def fill_panoptic_volume(volume, trackers, processes=4):
    r"""Fills a numpy or zarr array with the segmentations for all
    panoptic classes. Runs in-place.
    """
    for tracker in trackers:
        fill_volume(volume, tracker.instances, processes)

#----------------------------------------------------------
# Utilities for MultiGPU inference
#----------------------------------------------------------

def all_gather(tensor, group=None):
    f"""All gather operation for distributed multi-gpu group."""
    if not dist.is_available() or not dist.is_initialized():
        world_size = 1
    else:
        world_size = dist.get_world_size()

    # receiving Tensor from all ranks
    tensor_list = [
        torch.zeros_like(tensor) for _ in range(world_size)
    ]

    dist.all_gather(tensor_list, tensor, group=group)

    return tensor_list

def harden_seg(sem, confidence_thr):
    r"""Thresholds a binary segmentation or softmaxes
    and argmaxes a multiclass segmentation.
    """
    if sem.size(1) > 1: # multiclass segmentation
        sem = torch.argmax(sem, dim=1, keepdim=True)
    else:
        sem = (sem >= confidence_thr).long() # need integers not bool

    return sem

def get_panoptic_seg(
    sem,
    instance_cells,
    label_divisor,
    thing_list,
    stuff_area=32,
    void_label=0
):
    r"""Create pantopic segmentation from semantic segmentation
    and instance cells.
    """
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

def forward_multigpu(
    matchers,
    queue,
    rle_stack,
    matcher_in,
    confidence_thr,
    median_kernel_size,
    labels,
    label_divisor,
    thing_list,
    stuff_area=32,
    void_label=0
):
    r"""Handles segmentation median queue, panoptic postprocessing,
    conversion to run length encoded segmentation, and forward matching.
    Uses multiprocessing.Queue to receive predicted segmentations while
    model inference runs in parallel.
    """
    # create the queue for sem and instance cells
    median_queue = _MedianQueue(median_kernel_size)

    while True:
        sem, cells = queue.get()
        if isinstance(sem, str):
            # all images have been matched!
            break

        # update the queue
        median_queue.enqueue({'sem': sem, 'cells': cells})
        median_out = median_queue.get_next(keys=['sem'])

        # get segmentation if not None
        if median_out is not None:
            median_sem, cells = median_out['sem'], median_out['cells']
            median_sem = harden_seg(median_sem, confidence_thr)
            pan_seg = get_panoptic_seg(
                median_sem, cells, label_divisor,
                thing_list, stuff_area, void_label
            )
        else:
            continue

        # convert pan seg to rle
        pan_seg = pan_seg.squeeze().numpy()
        rle_seg = pan_seg_to_rle_seg(
            pan_seg, labels, label_divisor, thing_list, force_connected=True
        )

        # match the rle seg for each class
        rle_seg = apply_matchers(rle_seg, matchers)
        rle_stack.append(rle_seg)

    # get the final segmentations from the queue
    for qout in median_queue.end():
        sem, cells = qout['sem'], qout['cells']
        sem = harden_seg(sem, confidence_thr)
        pan_seg = get_panoptic_seg(
            sem, cells, label_divisor,
            thing_list, stuff_area, void_label
        )

        pan_seg = pan_seg.squeeze().numpy()
        rle_seg = pan_seg_to_rle_seg(
            pan_seg, labels, label_divisor, thing_list, force_connected=True
        )

        # match the rle seg for each class
        rle_seg = apply_matchers(rle_seg, matchers)
        rle_stack.append(rle_seg)

    matcher_in.send([rle_stack])
    matcher_in.close()
