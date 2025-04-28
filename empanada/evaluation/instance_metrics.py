import numpy as np

def f1(
    gt_matched,
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    r"""Calculates F1 score.

    Args:
        gt_matched: Array of object labels in ground truth that were matched
        at any IoU threshold.

        gt_unmatched: Array of object labels in ground truth that were
        unmatched.

        pred_matched: Array of object labels in prediction that were matched
        at any IoU threshold

        pred_unmatched: Array of object labels in prediction that were
        unmatched.

        matched_ious: Array of IoU scores for all pairs of matches between
        ground truth and prediction.

        iou_threshold: Float. The minimum IoU score between instances
        to count as a true positive.

    return:
        F1: Float. The F1@iou_threshold score.

    """
    # all unmatched gt are fn
    fn = len(gt_unmatched)
    # all unmatched pred are fp
    fp = len(pred_unmatched)

    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)

    # add 1 fp and 1 fn for every match that fails
    failed_matches = np.count_nonzero(matched_ious < iou_thr)
    fp += failed_matches
    fn += failed_matches

    if tp + fp + fn == 0:
        # by convention, F1 is 1 for empty masks
        return 1

    return tp / (tp + 0.5 * fp + 0.5 * fn)

def ap(
    gt_matched,
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    r"""Calculates average precision score.

    Args:
        gt_matched: Array of object labels in ground truth that were matched
        at any IoU threshold.

        gt_unmatched: Array of object labels in ground truth that were
        unmatched.

        pred_matched: Array of object labels in prediction that were matched
        at any IoU threshold

        pred_unmatched: Array of object labels in prediction that were
        unmatched.

        matched_ious: Array of IoU scores for all pairs of matches between
        ground truth and prediction.

        iou_threshold: Float. The minimum IoU score between instances
        to count as a true positive.

    return:
        ap: Float. The AP@iou_threshold score.

    """
    # all unmatched gt are fn
    fn = len(gt_unmatched)
    
    # all unmatched pred are fp
    fp = len(pred_unmatched)

    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)

    # add 1 fp and 1 fn for every match that fails
    failed_matches = np.count_nonzero(matched_ious < iou_thr)
    fp += failed_matches
    fn += failed_matches

    if tp + fp + fn == 0:
        # by convention, AP is 1 for empty masks
        return 1

    return tp / (tp + fp + fn)

def precision(
    gt_matched,
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    r"""Calculates precision.

    Args:
        gt_matched: Array of object labels in ground truth that were matched
        at any IoU threshold.

        gt_unmatched: Array of object labels in ground truth that were
        unmatched.

        pred_matched: Array of object labels in prediction that were matched
        at any IoU threshold

        pred_unmatched: Array of object labels in prediction that were
        unmatched.

        matched_ious: Array of IoU scores for all pairs of matches between
        ground truth and prediction.

        iou_threshold: Float. The minimum IoU score between instances
        to count as a true positive.

    return:
        Precision: Float. The detection precision

    """
    # all unmatched pred are fp
    fp = len(pred_unmatched)

    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)

    # add 1 fp for every match that failed
    fp += np.count_nonzero(matched_ious < iou_thr)

    if tp + fp == 0:
        # by convention, precision is 1 for empty masks
        return 1

    return tp / (tp + fp)

def recall(
    gt_matched,
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    r"""Calculates recall.

    Args:
        gt_matched: Array of object labels in ground truth that were matched
        at any IoU threshold.

        gt_unmatched: Array of object labels in ground truth that were
        unmatched.

        pred_matched: Array of object labels in prediction that were matched
        at any IoU threshold

        pred_unmatched: Array of object labels in prediction that were
        unmatched.

        matched_ious: Array of IoU scores for all pairs of matches between
        ground truth and prediction.

        iou_threshold: Float. The minimum IoU score between instances
        to count as a true positive.

    return:
        Recall: Float. The detection recall

    """
    # all unmatched gt are fn
    fn = len(gt_unmatched)

    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)

    # add 1 fn for every match that failed
    fn += np.count_nonzero(matched_ious < iou_thr)

    if tp + fn == 0:
        # by convention, recall is 1 for empty masks
        return 1

    return tp / (tp + fn)

def f1_50(**kwargs):
    return f1(**kwargs, iou_thr=0.5)

def f1_75(**kwargs):
    return f1(**kwargs, iou_thr=0.75)

def precision_50(**kwargs):
    return precision(**kwargs, iou_thr=0.5)

def precision_75(**kwargs):
    return precision(**kwargs, iou_thr=0.75)

def recall_50(**kwargs):
    return recall(**kwargs, iou_thr=0.5)

def recall_75(**kwargs):
    return recall(**kwargs, iou_thr=0.75)
