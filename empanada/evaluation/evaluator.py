import json
import numpy as np
from empanada.array_utils import *
from empanada.inference.matcher import rle_matcher

def _merge_encodings_for_semantic(pred_encodings):
    r"""Helper function to merge run length encodings of all instances
    for semantic evaluation
    """
    if len(pred_encodings) > 1:
        # convert to start and runs: (N, 2)
        pred_runs = np.concatenate(
            [np.stack(string_to_rle(enc), axis=1) for enc in pred_encodings]
        )

        # merge any overlapping runs
        rles = np.stack(merge_rles(pred_runs[:, 0], pred_runs[:, 1]), axis=1)
    else:
        rles = np.array([[-1, -1]])

    return rles

class Evaluator:
    r"""Evaluates model performance by comparing run length encoded
    3D segmentations.

    Args:
        semantic_metrics: Dicionary. Keys are metric names and values are
        semantic metric calculation functions.

        instance_metrics: Dicionary. Keys are metric names and values are
        instance metric calculation functions.

        panoptic_metrics: Dicionary. Keys are metric names and values are
        panoptic metric calculation functions.
    """
    def __init__(
        self,
        semantic_metrics=None,
        instance_metrics=None,
        panoptic_metrics=None
    ):
        self.semantic_metrics = semantic_metrics
        self.instance_metrics = instance_metrics
        self.panoptic_metrics = panoptic_metrics

    @staticmethod
    def _unpack_instance_dict(instance_dict):
        labels = []
        boxes = []
        encodings = []
        for k in instance_dict.keys():
            labels.append(int(k))
            boxes.append(instance_dict[k]['box'])
            encodings.append(instance_dict[k]['rle'])

        return np.array(labels), np.array(boxes), encodings

    def __call__(self, gt_json_fpath, pred_json_fpath, return_instances=False):
        # load the json files for each
        with open(gt_json_fpath, mode='r') as f:
            gt_json = json.load(f)

        with open(pred_json_fpath, mode='r') as f:
            pred_json = json.load(f)

        assert (gt_json['class_id'] == pred_json['class_id']), \
        "Prediction and ground truth classes must match!"

        gt_labels, gt_boxes, gt_encodings = self._unpack_instance_dict(gt_json['instances'])
        pred_labels, pred_boxes, pred_encodings = self._unpack_instance_dict(pred_json['instances'])

        semantic_results = {}
        instance_results = {}
        panoptic_results = {}

        if self.semantic_metrics is not None:
            # decode and concatenate all gt and pred encodings
            # N.B. This will break badly for dense semantic classes!
            gt_indices = np.concatenate([np.stack(string_to_rle(enc), axis=1) for enc in gt_encodings])
            pred_indices = _merge_encodings_for_semantic(pred_encodings)

            # calculate semantic metrics
            semantic_results = {name: func(gt_indices, pred_indices) for name,func in self.semantic_metrics.items()}

        if self.instance_metrics is not None or self.panoptic_metrics is not None:
            # match instances
            matched_labels, all_labels, matched_ious = \
            rle_matcher(gt_json['instances'], pred_json['instances'])

            gt_labels, gt_matched = all_labels[0], matched_labels[0]
            pred_labels, pred_matched = all_labels[1], matched_labels[1]

            # determine unmatched instance ids
            gt_unmatched = np.setdiff1d(gt_labels, gt_matched)
            pred_unmatched = np.setdiff1d(pred_labels, pred_matched)

            kwargs = {
                'gt_matched': gt_matched,
                'pred_matched': pred_matched,
                'gt_unmatched': gt_unmatched,
                'pred_unmatched': pred_unmatched,
                'matched_ious': matched_ious
            }

            if self.instance_metrics is not None:
                instance_results = {name: func(**kwargs) for name,func in self.instance_metrics.items()}

            if self.panoptic_metrics is not None:
                panoptic_results = {name: func(**kwargs) for name,func in self.panoptic_metrics.items()}

        # unpack all into 1 dictionary
        results_dict = {**semantic_results, **instance_results, **panoptic_results}
        if return_instances:
            instances_dict = {
                'gt_matched': gt_matched, 'pred_matched': pred_matched,
                'gt_unmatched': gt_unmatched, 'pred_unmatched': pred_unmatched,
                'matched_ious': matched_ious
            }
            return results_dict, instances_dict
        else:
            return results_dict
