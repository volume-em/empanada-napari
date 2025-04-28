"""
Copied and modified from:
https://github.com/bowenc0221/panoptic-deeplab/tree/master/segmentation/model/post_processing

Modifications:
--------------
1. Removed top k option from find_instance_center
2. Added chunking of instance labeling to avoid OOM errors
3. Miscellaneous formatting and docstrings.

"""

import torch
import torch.nn.functional as F
from typing import List

__all__ = [
    'factor_pad',
    'find_instance_center',
    'group_pixels',
    'get_instance_segmentation',
    'get_panoptic_segmentation'
]

@torch.jit.script
def factor_pad(tensor, factor: int=16):
    r"""Helper function to pad a tensor such that all dimensions are divisble
    by a particular factor
    """
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return F.pad(tensor, (0, pad_right, 0, pad_bottom))

@torch.jit.script
def find_instance_center(ctr_hmp, threshold: float=0.1, nms_kernel: int=7):
    r"""Find the center points from the center heatmap.

    Args:
        ctr_hmp: A Tensor of shape (N, 1, H, W) of raw center heatmap output, where N is the batch size,
        for consistent, we only support N=1.

        threshold: A Float, threshold applied to center heatmap score. Default 0.1.

        nms_kernel: An Integer, NMS max pooling kernel size. Default 7.

    Returns:
        ctr_all: A Tensor of shape (K, 2) where K is the number of center points. The order of second dim is (y, x).

    """
    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1.)

    # NMS
    nms_padding = nms_kernel // 2
    ctr_hmp_max_pooled = F.max_pool2d(
        ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding
    )

    if nms_kernel % 2 == 0:
        # clip last row and column to maintain size
        ctr_hmp_max_pooled = ctr_hmp_max_pooled[..., :-1, :-1]

    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1.

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, \
    'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    return ctr_all

@torch.jit.script
def chunked_pixel_grouping(ctr, ctr_loc, chunksize: int=20):
    r"""Gives each pixel in the image an instance id without exceeding memory.

    Args:
        ctr: A Tensor of shape [K, 1, 2] where K is the number of center points.
        The order of third dim is (y, x).

        ctr_loc: A Tensor of shape [1, H*W, 2] of center locations for each pixel
        after applying offsets.

        chunksize: Int. Number of instances to process in a chunk. Default 20.

    Returns:
        instance_ids: A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).

    """
    # initialize a tensor to store nearest instance center ids
    # and a distances placeholder of large floats
    instance_ids = torch.zeros(ctr_loc.size(1), dtype=torch.long, device=ctr.device) # (H*W,)
    nearest_distances = 1e5 * torch.ones(ctr_loc.size(1), dtype=torch.float, device=ctr.device) # (H*W,)

    # split the centers into chunks
    ctr_chunks = torch.split(ctr, chunksize, dim=0)
    prev = 1 # starting label for instance ids

    for ctr_chunk in ctr_chunks:
        # chunk of size (chunksize, 1, 2)
        distances = torch.norm(ctr_chunk - ctr_loc, dim=-1) # (chunksize, H*W)
        min_distances, min_dist_indices = distances.min(dim=0) # (H*W,)

        # add the instance ids relative to the previous label
        instance_ids[min_distances < nearest_distances] = prev + min_dist_indices[min_distances < nearest_distances]
        nearest_distances = torch.min(nearest_distances, min_distances)

        # increment the instance ids
        prev += ctr_chunk.size(0)

    return instance_ids

@torch.jit.script
def group_pixels(ctr, offsets, chunksize: int=20, step: float=1):
    r"""
    Gives each pixel in the image an instance id.

    Args:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).

        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size of 1.
        The order of second and third dim is (offset_y, offset_x).

        chunksize: Int. Number of instances to process in a chunk. Default 20.

    Returns:
        instance_ids: A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).

    """
    assert ctr.size(0) > 0
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    hend = int(height * step)
    wend = int(width * step)

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(0, hend, step=step, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(0, wend, step=step, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    # multiply the ctrs by step
    ctr = step * ctr

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    if ctr.size(0) <= chunksize:
        distance = torch.norm(ctr - ctr_loc, dim=-1) # (K, H*W)
        instance_id = 1 + torch.argmin(distance, dim=0) # (H*W)
    else:
        instance_id = chunked_pixel_grouping(ctr, ctr_loc, chunksize)

    instance_id = instance_id.reshape((1, height, width))

    return instance_id

@torch.jit.script
def get_instance_segmentation(
    sem_seg,
    ctr_hmp,
    offsets,
    thing_list: List[int],
    threshold: float=0.1,
    nms_kernel: int=7
):
    r"""Post-processing for instance segmentation, gets class agnostic instance id map.

    Args:
        sem_seg: A Tensor of shape (N, 1, H, W), predicted semantic label.

        ctr_hmp: A Tensor of shape (N, 1, H, W) of raw center heatmap output,
        where N is the batch size of 1.

        offsets: A Tensor of shape (N, 2, H, W) of raw offset output.
        The order of second dim is (offset_y, offset_x).

        thing_list: A List of instance class ids.

        threshold: A Float, threshold applied to center heatmap score. Default 0.1.

        nms_kernel: An Integer, NMS max pooling kernel size. Default 7.

    Returns:
        thing_seg: A Tensor of shape (1, H, W).

        ctr: A Tensor of shape (1, K, 2) where K is the number of center points.
        The order of second dim is (y, x).

    """
    assert sem_seg.size(0) == 1, \
    f'Only batch size of 1 is supported!'

    sem_seg = sem_seg[0]

    # keep only label for instance classes
    instance_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        instance_seg[sem_seg == thing_class] = 1

    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel)

    # no objects, return zeros
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)

    instance_id = group_pixels(ctr, offsets)
    return instance_seg * instance_id, ctr.unsqueeze(0)

@torch.jit.script
def merge_semantic_and_instance(
    sem_seg,
    ins_seg,
    label_divisor: int,
    thing_list: List[int],
    stuff_area: int,
    void_label: int
):
    r"""Post-processing for panoptic segmentation, by merging semantic
    segmentation label and class agnostic instance segmentation label.

    Args:
        sem_seg: A Tensor of shape (1, H, W), predicted semantic label.

        ins_seg: A Tensor of shape (1, H, W), predicted instance label.

        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.

        thing_list: A List of thing class ids.

        stuff_area: An Integer, remove stuff whose area is less than stuff_area.

        void_label: An Integer, indicates the region has no confident prediction.

    Returns:
        merged_seg: A Tensor of shape (1, H, W).

    """
    # In case thing mask does not align with semantic prediction
    pan_seg = torch.zeros_like(sem_seg) + void_label
    thing_seg = ins_seg > 0
    semantic_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg == thing_class] = 1

    # keep track of instance id for each class
    class_id_tracker: Dict[int, int] = {}

    # paste thing by majority voting
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue

        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (ins_seg == ins_id) & (semantic_thing_seg == 1)
        if torch.count_nonzero(thing_mask) == 0:
            continue

        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1, ))
        if class_id.item() in class_id_tracker:
            new_ins_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_ins_id = 1

        class_id_tracker[class_id.item()] += 1
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # paste stuff to unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            continue

        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)

        if area >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg

@torch.jit.script
def get_panoptic_segmentation(
    sem,
    ctr_hmp,
    offsets,
    thing_list: List[int],
    label_divisor: int,
    stuff_area: int,
    void_label: int,
    threshold: float=0.1,
    nms_kernel: int=7
):
    r"""Post-processing for panoptic segmentation.

    Args:
        sem_seg: A Tensor of shape (N, 1, H, W), predicted semantic labels.

        ctr_hmp: A Tensor of shape (N, 1, H, W) of raw center heatmap output,
        where N is the batch size of 1.

        offsets: A Tensor of shape (N, 2, H, W) of raw offset output.
        The order of second dim is (offset_y, offset_x).

        thing_list: A List of thing class ids.

        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.

        stuff_area: An Integer, remove stuff whose area is less than stuff_area.

        void_label: An Integer, indicates the region has no confident prediction.

        threshold: A Float, threshold applied to center heatmap score. Default 0.1.

        nms_kernel: An Integer, NMS max pooling kernel size. Default 7.

    Returns:
        pan_seg: A Tensor of shape (1, H, W) of type torch.long.

    """

    if sem.size(1) != 1:
        raise ValueError('Expect single channel semantic segmentation. Softmax/argmax first!')
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # instance segmentation with thing centers
    instance, center = get_instance_segmentation(
        sem, ctr_hmp, offsets, thing_list, threshold=threshold, nms_kernel=nms_kernel
    )

    panoptic = merge_semantic_and_instance(
        sem, instance, label_divisor, thing_list, stuff_area, void_label
    )

    return panoptic, center
