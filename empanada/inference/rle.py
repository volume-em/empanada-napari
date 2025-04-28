import sys
import numpy as np
from empanada.array_utils import *
from skimage import measure

# optional import for faster connected components
try:
    import cc3d
except:
    pass

__all__ = [
    'pan_seg_to_rle_seg',
    'rle_seg_to_pan_seg',
    'unpack_rle_attrs'
]

def connected_components(seg):
    if 'cc3d' in sys.modules:
        seg = cc3d.connected_components(seg, connectivity=8, out_dtype=np.uint32)
    else:
        seg = measure.label(seg)

    return seg

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
        instance_attrs = {}
        rps = measure.regionprops(instance_seg)
        for rp in rps:
            # convert from label xy coords to rles
            coords_flat = np.ravel_multi_index(tuple(rp.coords.T), instance_seg.shape)
            starts, runs = rle_encode(coords_flat)

            instance_attrs[rp.label] = {'box': rp.bbox, 'starts': starts, 'runs': runs}

        # add to the rle_seg
        rle_seg[label] = instance_attrs

    return rle_seg

def rle_seg_to_pan_seg(
    rle_seg,
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

    for instance_attrs in rle_seg.values():
        for object_id, attrs in instance_attrs.items():
            starts = attrs['starts']
            runs = attrs['runs']

            for s,r in zip(starts, runs):
                pan_seg[s:s+r] = object_id

    return pan_seg.reshape(shape)

def unpack_rle_attrs(instance_rle_seg):
    r"""Helper function to unpack the rle dictionary
    to a few lists.

    Args:
        instance_rle_seg: Dictionary of all instances for a given class_id.

    Returns:
        labels: Array of (n,). All labels in the instance_rle_seg.
        boxes: Array of (n, 4). Bounding boxes for each instance.
        starts: Array of (l,). Starts for each run.
        runs: Array of (l,). Run length for each run.

    """
    # extract labels, boxes, and rle for a given class id
    labels = []
    boxes = []
    starts = []
    runs = []
    for label,attrs in instance_rle_seg.items():
        labels.append(int(label))
        boxes.append(attrs['box'])
        if 'rle' in attrs:
            rle = string_to_rle(attrs['rle'])
            starts.append(rle[0])
            runs.append(rle[1])
        else:
            starts.append(attrs['starts'])
            runs.append(attrs['runs'])

    return np.array(labels), np.array(boxes), starts, runs
