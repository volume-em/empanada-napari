import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from empanada.inference.matcher import fast_matcher

__all__ = [
    'EMAMeter',
    'AverageMeter',
    'IoU', 'PQ', 'F1',
    'ComposeMetrics'
]

class EMAMeter:
    r"""Computes and stores an exponential moving average and current value"""
    def __init__(self, momentum=0.98):
        self.mom = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = (self.sum * self.mom) + (val * (1 - self.mom))
        self.count += 1
        self.avg = self.sum / (1 - self.mom ** (self.count))

class AverageMeter:
    r"""Computes and stores a moving average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count

class _BaseMetric:
    r"""Default Metric class"""
    def __init__(self, meter, labels):
        self.meters = {l: meter() for l in labels}
        self.labels = labels

    def update(self, value_dict):
        for l,v in value_dict.items():
            self.meters[l].update(v)

    def reset(self):
        for l in self.labels:
            self.meters[l].reset()

    def average(self):
        return {l: meter.avg for l,meter in self.meters.items()}

class IoU(_BaseMetric):
    r"""Computes the IoU between output and target.
    Input is expected to be a dictionary for each.

    Args:
        meter: EMAMeter or AverageMeter to track.
        labels: List of all semantic/instance labels to compare.
        output_key: Key in the output dictionary to compare.
        target_key: Key in the target dictionary to compare.

    """
    def __init__(
        self,
        meter,
        labels,
        output_key='sem_logits',
        target_key='sem',
        **kwargs
    ):
        super().__init__(meter, labels)
        self.output_key = output_key
        self.target_key = target_key

    def calculate(self, output, target):
        # only require the semantic segmentation
        output = output[self.output_key]
        target = target[self.target_key]

        # make target the same shape as output by unsqueezing
        # the channel dimension, if needed
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)

        # get the number of classes from the output channels
        n_classes = output.size(1)

        # get reshape size based on number of dimensions
        # can exclude first 2 dims, which are always batch and channel
        empty_dims = (1,) * (target.ndim - 2)

        if n_classes > 1:
            # one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)

            # softmax the output
            output = nn.Softmax(dim=1)(output)

            # binarize to most likely class
            max_probs = torch.max(output, dim=1, keepdim=True)[0]
            output = (output == max_probs).long()
        else:
            # sigmoid the output and binarize
            output = (torch.sigmoid(output) > 0.5).long()

        # cast target to the correct type
        target = target.type(output.dtype)

        # multiply the tensors, everything that is still as 1 is part of the intersection
        # (N,)
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersect = torch.sum(output * target, dims)

        # compute the union, (N,)
        union = torch.sum(output, dims) + torch.sum(target, dims) - intersect

        # avoid division errors by adding a small epsilon
        # evaluates to iou of 1 when intersect and union are 0
        iou = (intersect + 1e-5) / (union + 1e-5)

        if n_classes == 1:
            return {self.labels[0]: iou.item()}
        else:
            return {l: iou[l] for l in self.labels}

class PQ(_BaseMetric):
    r"""Computes the panoptic quality between output and target.
    Input is expected to be a dictionary for each.

    Args:
        meter: EMAMeter or AverageMeter to track
        labels: List of all semantic/instance labels to compare
        label_divisor: Integer. Label divisor used during postprocessing.
        output_key: Key in the output dictionary to compare.
        target_key: Key in the target dictionary to compare.

    """
    def __init__(
        self,
        meter,
        labels,
        label_divisor,
        output_key='pan_seg',
        target_key='pan_seg',
        **kwargs
    ):
        super().__init__(meter, labels)
        self.label_divisor = label_divisor
        self.output_key = output_key
        self.target_key = target_key

    def _to_class_seg(self, pan_seg, label):
        instance_seg = np.copy(pan_seg) # copy for safety
        min_id = label * self.label_divisor
        max_id = min_id + self.label_divisor

        # zero all objects/semantic segs outside of instance_id range
        outside_mask = np.logical_or(instance_seg < min_id, instance_seg >= max_id)
        instance_seg[outside_mask] = 0
        return instance_seg

    def calculate(self, output, target):
        # convert tensors to numpy
        output = output[self.output_key].squeeze().long().cpu().numpy()
        target = target[self.target_key].squeeze().long().cpu().numpy()

        # compute the panoptic quality, per class
        per_class_results = {}
        for label in self.labels:
            pred_class_seg = self._to_class_seg(output, label)
            tgt_class_seg = self._to_class_seg(target, label)

            # match the segmentations
            matched_labels, all_labels, matched_ious = \
            fast_matcher(tgt_class_seg, pred_class_seg, iou_thr=0.5)

            tp = len(matched_labels[0])
            fn = len(np.setdiff1d(all_labels[0], matched_labels[0]))
            fp = len(np.setdiff1d(all_labels[1], matched_labels[1]))

            if tp + fp + fn == 0:
                # by convention, PQ is 1 for empty masks
                per_class_results[label] = 1.
                continue

            sq = matched_ious.sum() / (tp + 1e-5)
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = sq * rq

        return per_class_results

class F1(_BaseMetric):
    r"""Computes the F1 between output and target instance segmentation
    classes. Input is expected to be a dictionary for each.

    Args:
        meter: EMAMeter or AverageMeter to track
        labels: List of all instance labels to compare
        label_divisor: Integer. Label divisor used during postprocessing.
        iou_thr: Float, IoU threshold at which to determine TP, FP, FN detections.
        output_key: Key in the output dictionary to compare.
        target_key: Key in the target dictionary to compare.

    """
    def __init__(
        self,
        meter,
        labels,
        label_divisor,
        iou_thr=0.5,
        output_key='pan_seg',
        target_key='pan_seg',
        **kwargs
    ):
        super().__init__(meter, labels)
        self.label_divisor = label_divisor
        self.iou_thr = iou_thr
        self.output_key = output_key
        self.target_key = target_key

    def _to_class_seg(self, pan_seg, label):
        instance_seg = np.copy(pan_seg) # copy for safety
        min_id = label * self.label_divisor
        max_id = min_id + self.label_divisor

        # zero all objects/semantic segs outside of instance_id range
        outside_mask = np.logical_or(instance_seg < min_id, instance_seg >= max_id)
        instance_seg[outside_mask] = 0
        return instance_seg

    def calculate(self, output, target):
        # convert tensors to numpy
        output = output[self.output_key].squeeze().long().cpu().numpy()
        target = target[self.target_key].squeeze().long().cpu().numpy()

        # compute the panoptic quality, per class
        per_class_results = {}
        for label in self.labels:
            pred_class_seg = self._to_class_seg(output, label)
            tgt_class_seg = self._to_class_seg(target, label)

            # match the segmentations
            matched_labels, all_labels, matched_ious = \
            fast_matcher(tgt_class_seg, pred_class_seg, iou_thr=self.iou_thr)

            tp = len(matched_labels[0])
            fn = len(np.setdiff1d(all_labels[0], matched_labels[0]))
            fp = len(np.setdiff1d(all_labels[1], matched_labels[1]))

            if tp + fp + fn == 0:
                # by convention, F1 is 1 for empty masks
                per_class_results[label] = 1.
            else:
                f1 = tp / (tp + 0.5 * fn + 0.5 * fp)
                per_class_results[label] = f1

        return per_class_results

class ComposeMetrics:
    r"""Bundles multiple metrics together for easy
    evaluation, printing and logging during training.

    Args:
        metrics_dict: Dictionary, keys are the names of metrics and values are
            the _BaseMetric class than records/calculate that metric.

        class_names: Dictionary, keys are class_ids and values are names.

        reset_on_print: Bool. If True, the history of each metric is wiped
            after results are printed.

    """
    def __init__(
        self,
        metrics_dict,
        class_names,
        reset_on_print=True
    ):
        self.metrics_dict = metrics_dict
        self.class_names = class_names
        self.reset_on_print = reset_on_print
        self.history = {}

    def evaluate(self, output, target):
        # calculate all the metrics in the dict
        for metric in self.metrics_dict.values():
            value = metric.calculate(output, target)
            metric.update(value)

    def display(self):
        print_names = []
        print_values = []
        for metric_name, metric in self.metrics_dict.items():
            avg_values = metric.average()

            for l, v in avg_values.items():
                value_name = self.class_names[l]
                print_values.append(float(v))
                print_names.append(f'{value_name}_{metric_name}')

            if self.reset_on_print:
                metric.reset()

        # print out the metrics
        for name, value in zip(print_names, print_values):
            if name not in self.history:
                self.history[name] = [value]
            else:
                self.history[name].append(value)

            print(name, value)
