import numpy as np
from empanada.array_utils import rle_iou

def iou(
    gt_rle,
    pred_rle
):
    r"""Calculates IoU between semantic run length encodings.

    Args:
        gt_rle: Array of (n, 2) where each element defines a run of
        (start, run_length).

        pred_rle: Array of (m, 2) where each element defines a run of
        (start, run_length).

    return:
        IoU: Float. The intersection-over-union score.

    """
    if len(gt_rle) == 0 and len(pred_rle) == 0:
        return 1
    elif len(gt_rle) == 0 or len(pred_rle) == 0:
        return 0

    return rle_iou(gt_rle[:, 0], gt_rle[:, 1], pred_rle[:, 0], pred_rle[:, 1])
