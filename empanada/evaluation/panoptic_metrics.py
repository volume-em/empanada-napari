import numpy as np

def panoptic_quality(
    gt_matched,
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious
):
    r"""Calculates panoptic quality.

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

    return:
        PQ: Float. The panoptic quality score.

    """
    # all unmatched gt are fn
    fn = len(gt_unmatched)
    # all unmatched pred are fp
    fp = len(pred_unmatched)

    # matches can be tp, fp or fn depending
    # on the iou score
    tp_ious = matched_ious[matched_ious >= 0.5]
    tp = len(tp_ious)

    # add 1 fp and 1 fn for every match that fails
    failed_matches = np.count_nonzero(matched_ious < 0.5)
    fp += failed_matches
    fn += failed_matches

    if tp + fp + fn == 0:
        # by convention, PQ is 1 for empty masks
        return 1

    sq = tp_ious.sum() / (tp + 1e-5)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)

    return sq * rq
