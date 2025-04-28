"""
Copied and modified from: https://github.com/zudi-lin/pytorch_connectomics/blob/b6457ea4bc7d9b01ef3a00781dff252ab5d4a1d3/connectomics/utils/process.py
"""

import sys
import heapq
import numba
import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.morphology._util import _validate_connectivity, _offsets_to_raveled_neighbors
from skimage.util import crop

# optional import for faster connected components
try:
    import cc3d
except:
    pass

__all__ = ['bc_watershed']

def connected_components(seg):
    if 'cc3d' in sys.modules:
        return cc3d.connected_components(seg, connectivity=26, out_dtype=np.uint32)
    else:
        return label(seg).astype(np.uint32)
    
def size_threshold(seg, thres):
    if 'cc3d' in sys.modules:
        return cc3d.dust(seg, threshold=thres, connectivity=26)
    else:
        return remove_small_objects(seg, thres)

def cast2dtype(segm):
    r"""Cast the segmentation mask to the best dtype to save storage.
    """
    mid = np.max(segm)

    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32

    return segm.astype(m_type)

@numba.jit
def _mask_watershed_jit(mask, marker_locations, flat_neighborhood, output):
    r"""A simplified implementation of the watershed algorithm that assumes
    the watershed image is a binary labelmap.

    Args:
        mask (np.ndarray): A raveled binary segmentation, either bool or uint8
        marker_locations (np.ndarray of ints): Indices in the image that
        correspond to the watershed markers.
        flat_neighborhood (np.ndarray): Index offsets to lookup nearest neighbors.
        output (np.ndarray): The watershed output array. Modified inplace.

    """
    age = 0
    h = [(0, 0)]
    for ml in marker_locations:
        heapq.heappush(h, (age, ml)) # only need age and index

    # get rid of typing seed (0, 0)
    heapq.heappop(h)

    while h:
        elem = heapq.heappop(h)

        age += 1
        for n_offset in flat_neighborhood:
            neighbor_index = n_offset + elem[1]

            if not mask[neighbor_index]:
                continue

            if output[neighbor_index]:
                continue

            output[neighbor_index] = output[elem[1]]
            heapq.heappush(h, (age, neighbor_index))

def mask_watershed(mask, markers, connectivity=1):
    r"""A simplified implementation of the watershed algorithm that assumes
    the watershed image is a binary labelmap.

    Args:
        mask (np.ndarray): A binary segmentation, either bool or uint8
        marker (np.ndarray): Labelmap of seeds for watershed.
        connectivity (int or np.ndarray): 1 or 2 connectivity or a boolean array
        defining neighboor of each point.

    Returns:
        watershed_seg (np.ndarray of marker.dtype): The segmentation after
        applying watershed.

    """
    # helper from skimage
    connectivity, offset = _validate_connectivity(
        mask.ndim, connectivity, offset=None
    )

    # pad and ravel mask
    pad_width = [(p, p) for p in offset]
    mask = np.pad(mask, pad_width, mode='constant')
    output = np.pad(markers, pad_width, mode='constant')

    flat_neighborhood = _offsets_to_raveled_neighbors(
        mask.shape, connectivity, center=offset
    )

    marker_locations = np.flatnonzero(output)

    _mask_watershed_jit(
        mask.ravel(), marker_locations,
        flat_neighborhood, output.ravel()
    )

    output = crop(output, pad_width, copy=True)

    return output

def bc_watershed(
    volume,
    thres1=0.9,
    thres2=0.8,
    thres3=0.85,
    seed_thres=32,
    min_size=128,
    label_divisor=1000,
    use_mask_wts=False
):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via watershed segmentation algorithm.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape (C, Z, Y, X).
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        seed_thr (int): minimum size of seed in voxels. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        use_mask_wts (bool): whether to use mask watershed for memory and (and sometimes) speed.
    """

    assert volume.shape[0] == 2
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
    foreground = (semantic > int(255*thres3))
    
    # prepare seeds
    seed = connected_components(seed_map)
    seed = size_threshold(seed, seed_thres)
    
    if use_mask_wts:
        segm = mask_watershed(foreground, seed)
    else:
        segm = watershed(-semantic.astype(np.float64), seed, mask=foreground).astype(np.uint32)

    if min_size is not None:
        segm = size_threshold(segm, min_size)
        
    segm[segm > 0] += label_divisor

    return cast2dtype(segm)
