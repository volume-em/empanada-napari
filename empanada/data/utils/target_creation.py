import cv2
import numpy as np
import torch
from skimage import measure
from skimage.morphology import dilation
from scipy.signal import convolve2d

__all__ = [
    'heatmap_and_offsets',
    'seg_to_instance_bd'
]

def heatmap_and_offsets(sl2d, heatmap_sigma=6):
    r"""Creates center heatmap and offsets for panoptic deeplab
    training.

    Args:
        sl2d: Array of (h, w) defining an instance segmentation.

        heatmap_sigma: Float. Standard deviation of the Guassian filter
        that is applied to create the center heatmap.

    Returns:
        heatmap: Array of (1, h, w) defining the heatmap of instance centers.

        offsets: Array of (2, h, w) defining the offsets from each pixel
        to the associated instance center. Channels are up-down and
        left-right offsets respectively.

    """
    # make sure, the input is numpy
    convert = False
    if type(sl2d) == torch.Tensor:
        sl2d = sl2d.numpy()
        convert = True

    h, w = sl2d.shape
    centers = np.zeros((2, h, w), dtype=np.float32)
    heatmap = np.zeros((h, w), dtype=np.float32)

    # loop over the instance labels and store
    # relevant centers for each
    rp = measure.regionprops(sl2d)
    for r in rp:
        sub_label = r.label
        y, x = r.centroid
        heatmap[int(y), int(x)] = 1
        centers[0, sl2d == sub_label] = y
        centers[1, sl2d == sub_label] = x

    # apply a gaussian filter to spread the centers
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0),
                               sigmaX=heatmap_sigma, sigmaY=heatmap_sigma,
                               borderType=cv2.BORDER_CONSTANT)

    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax

    # convert from centers to offsets
    yindices = np.arange(0, h, dtype=np.float32)
    xindices = np.arange(0, w, dtype=np.float32)

    # add the y indices to the first channel
    # in the output and x indices to the second channel
    offsets = np.zeros_like(centers)
    offsets[0] = centers[0] - yindices[:, None]
    offsets[1] = centers[1] - xindices[None, :]
    offsets[:, sl2d == 0] = 0

    # add empty dimension to heatmap
    heatmap = heatmap[None] # (1, H, W)

    if convert:
        heatmap = torch.from_numpy(heatmap)
        offsets = torch.from_numpy(offsets)

    return heatmap, offsets

def seg_to_instance_bd(
    seg: np.ndarray,
    tsz_h: int = 1,
    do_bg: bool = True,
    do_convolve: bool = True
) -> np.ndarray:
    r"""Generate instance contour map from segmentation masks.

    Args:
        seg (np.ndarray): segmentation map (3D array is required).
        tsz_h (int, optional): size of the dilation struct. Defaults: 1
        do_bg (bool, optional): generate contour between instances and background. Defaults: True
        do_convolve (bool, optional): convolve with edge filters. Defaults: True
    Returns:
        np.ndarray: binary instance contour map.

    Copied from: https://github.com/zudi-lin/pytorch_connectomics/blob/72d6a0fc75a3275f79fa96c90605abb814bd7a97/connectomics/data/utils/data_segmentation.py

    """
    if do_bg == False:
        do_convolve = False

    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    tsz = tsz_h * 2 + 1

    if do_convolve:
        sobel = [1, 0, -1]
        sobel_x = np.array(sobel).reshape(3, 1)
        sobel_y = np.array(sobel).reshape(1, 3)
        for z in range(sz[0]):
            slide = seg[z]
            edge_x = convolve2d(slide, sobel_x, 'same', boundary='symm')
            edge_y = convolve2d(slide, sobel_y, 'same', boundary='symm')
            edge = np.maximum(np.abs(edge_x), np.abs(edge_y))
            contour = (edge != 0).astype(np.uint8)
            bd[z] = dilation(contour, np.ones((tsz, tsz), dtype=np.uint8))

        return bd

    mm = seg.max()
    for z in range(sz[0]):
        patch = im2col(
            np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        if do_bg:  # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0 > 0)*(p0 != p1)).reshape(sz[1:])
        else:  # between two non-zero seg
            patch[patch == 0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0 != 0)*(p1 != 0)*(p0 != p1)).reshape(sz[1:])

    return bd
