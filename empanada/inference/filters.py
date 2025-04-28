import numpy as np
from empanada.array_utils import *
from skimage.morphology import erosion, dilation
from skimage import measure
from scipy.ndimage import binary_fill_holes
import sys
import tqdm

__all__ = [
    'remove_small_objects',
    'remove_pancakes'
]

def connected_components(seg):
    if 'cc3d' in sys.modules:
        seg = cc3d.connected_components(seg, connectivity=8, out_dtype=np.uint32)
    else:
        seg = measure.label(seg)

    return seg

def remove_small_objects(object_tracker, min_size=64):
    r"""Deletes small objects from an object tracker in-place

    Args:
        object_tracker: empanda.inference.trackers.InstanceTracker
        min_size: Integer, minimum size of object in voxels.
    """
    instance_ids = list(object_tracker.instances.keys())
    for instance_id in instance_ids:
        # sum all the runs in instance
        instance_attrs = object_tracker.instances[instance_id]
        size = instance_attrs['runs'].sum()

        if size < min_size:
            del object_tracker.instances[instance_id]

def remove_pancakes(object_tracker, min_span=4):
    r"""Deletes pancake-shaped objects from an object tracker in-place

    Args:
        object_tracker: empanda.inference.trackers.InstanceTracker
        min_span: Integer, the minimum extent of the objects bounding box.
    """
    instance_ids = list(object_tracker.instances.keys())
    for instance_id in instance_ids:
        # load the box of the instance
        instance_attrs = object_tracker.instances[instance_id]
        box = instance_attrs['box']

        zspan = box[3] - box[0]
        yspan = box[4] - box[1]
        xspan = box[5] - box[2]

        if any(span < min_span for span in [zspan, yspan, xspan]):
            del object_tracker.instances[instance_id]


def pan_seg_to_rle_seg(
    pan_seg,
    labels,
    label_divisor,
    thing_list,
    force_connected=True
):
    r"""Converts a panoptic segmentation to run length encodings.

    Args:
        pan_seg: Array of (h, w) defining a panoptic segmentation.

        labels: List of integers. All labels from pan_seg to encode.

        label_divisor: Integer. The label divisor used to postprocess
        the panoptic segmentation.

        thing_list: List of integers. All class_id in labels that are
        instance classses.

        force_connected: Whether to enforce that instances be
        connected components.

    Returns:
        rle_seg: Nested dictionary. Top level keys are 'labels', values is
        a dictionary. Keys in this second level are 'instance_ids', values
        is a dictionary. Keys in this last level are 'box', 'starts', 'runs'
        that define the extents and run length encoding of the instance.

    """
    # convert from dense panoptic seg to sparse rle segment class
    rle_seg = {}
    instance_attrs = {}
    for label in labels:
        # convert from pan_seg to instance_seg
        min_id = label * label_divisor
        max_id = min_id + label_divisor

        # zero all objects/semantic segs outside of instance_id range
        instance_seg = pan_seg.copy()
        outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
        instance_seg[outside_mask] = 0

        # relabel connected components
        if force_connected and label in thing_list:
            instance_seg = connected_components(instance_seg)
            instance_seg[instance_seg > 0] += min_id

        # measure the regionprops

        rps = measure.regionprops(instance_seg)
        for rp in rps:
            # convert from label xy coords to rles
            coords_flat = np.ravel_multi_index(tuple(rp.coords.T), instance_seg.shape)
            starts, runs = rle_encode(coords_flat)

            instance_attrs[rp.label] = {'box': rp.bbox, 'starts': starts, 'runs': runs}

        # add to the rle_seg
        rle_seg[label] = instance_attrs

    return instance_attrs

def rle_seg_to_pan_seg(
    object_tracker,
    shape
):
    r"""Converts run length encodings to a panoptic segmentation.

    Args:
        rle_seg: Nested dictionary. Output of pan_seg_to_rle_seg function.
        Top level keys are 'labels', values is a dictionary. Keys in this
        second level are 'instance_ids', values is a dictionary. Keys in this
        last level are 'box', 'starts', 'runs' that define the extents and
        run length encoding of the instance.

        shape: Tuple of integers. The (height, width) of the pan_seg.

    Returns:
        pan_seg: Array of (h, w) defining a panoptic segmentation.

    """
    # convert from dense panoptic seg to sparse rle segment class
    pan_seg = np.zeros(shape, dtype=np.uint32).ravel()

    for object_id, attrs in object_tracker.instances.items():
        starts = attrs['starts']
        runs = attrs['runs']

        for s,r in zip(starts, runs):
            pan_seg[s:s+r] = object_id

    return pan_seg.reshape(shape)


def erode(object_tracker, volume_shape, labels, label_divisor, thing_list, iterations=1):

    mask = rle_seg_to_pan_seg(object_tracker, volume_shape)

    for _ in tqdm.tqdm(range(iterations), desc='eroding labels:'):
        mask = erosion(mask)

    object_tracker.instances = pan_seg_to_rle_seg(mask, labels, label_divisor, thing_list)

    return object_tracker

def dilate(object_tracker, volume_shape, labels, label_divisor, thing_list, iterations=1):
    mask = rle_seg_to_pan_seg(object_tracker, volume_shape)

    for _ in tqdm.tqdm(range(iterations), desc='dilating labels:'):
        mask = dilation(mask)

    object_tracker.instances = pan_seg_to_rle_seg(mask, labels, label_divisor, thing_list)

    return object_tracker

def fill_holes_in_segmentation(object_tracker, volume_shape, labels, label_divisor, thing_list):
    r"""Fills holes in the segmentation of an object tracker in-place using ndimage binary fill holes

    Args:
        object_tracker: empanda.inference.trackers.InstanceTracker
        volume_shape: Tuple of integers. The (height, width) of the volume.
        labels: List of integers. All labels from pan_seg to encode.
        label_divisor: Integer. The label divisor used to postprocess
        the panoptic segmentation.
        thing_list: List of integers. All class_id in labels that are
        instance classses.
    """
    mask_3d = rle_seg_to_pan_seg(object_tracker, volume_shape)

    if mask_3d.ndim == 3:
        for idx in tqdm.tqdm(range(mask_3d.shape[0]), desc='filling holes in labels:'):
            mask = mask_3d[idx]
            # fill holes in the mask
            unique_indices = np.unique(mask)
            rprops = measure.regionprops(mask)

            # crop labels and then apply fill holes
            for rp in rprops:
                if rp.label in unique_indices and rp.label > 0:
                    minr, minc, maxr, maxc = rp.bbox

                    tmp = mask[minr:maxr, minc:maxc]
                    tmp = binary_fill_holes(tmp.astype(bool))
                    mask[minr:maxr, minc:maxc] = tmp.astype(mask.dtype)*rp.label
            mask_3d[idx] = mask
    else:
        print('mask_3d is not 3D')
        mask = mask_3d

    object_tracker.instances = pan_seg_to_rle_seg(mask_3d, labels, label_divisor, thing_list)
    return object_tracker


