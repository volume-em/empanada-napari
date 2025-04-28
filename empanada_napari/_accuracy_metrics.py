#!/usr/bin/env python3
import argparse
import os
import numpy as np
import cv2
from magicgui import magicgui
from napari.layers import Image, Labels
from scipy.optimize import linear_sum_assignment

def compute_pixel_metrics(pred, gt):
    """
    Computes pixel-level metrics between the predicted and ground truth label images.
    First, the images are binarized (using threshold > 0).

    Metrics computed:
      - Overall pixel accuracy.
      - Per-label accuracy.
      - Mean Intersection over Union (mIoU).
      - Mean Dice coefficient.
    """
    if pred.shape != gt.shape:
        raise ValueError("The shape of the prediction and ground truth images must match.")

    # Binarize the images: any value > 0 is set to 1.
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # Compute overall accuracy.
    overall_accuracy = np.mean(pred == gt)

    # Since the images are now binary, we only have two labels: 0 (background) and 1 (foreground).
    labels = [0, 1]

    iou_list = []
    dice_list = []
    per_label_acc = {}

    for label in labels:
        pred_mask = (pred == label)
        gt_mask = (gt == label)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union != 0 else np.nan

        dice_denom = pred_mask.sum() + gt_mask.sum()
        dice = (2 * intersection) / dice_denom if dice_denom != 0 else np.nan

        iou_list.append(iou)
        dice_list.append(dice)

        # Compute accuracy for this label as correctly predicted pixels / total ground truth pixels.
        gt_total = gt_mask.sum()
        acc = intersection / gt_total if gt_total else np.nan
        per_label_acc[label] = acc

    mean_iou = np.nanmean(iou_list)
    mean_dice = np.nanmean(dice_list)

    return overall_accuracy, per_label_acc, mean_iou, mean_dice


def compute_instance_iou_dice(pred_mask, gt_mask):
    """
    Computes the Intersection over Union (IoU) and Dice coefficient between two binary masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0.0
    dice_denom = pred_mask.sum() + gt_mask.sum()
    dice = (2 * intersection) / dice_denom if dice_denom != 0 else 0.0
    return iou, dice


def compute_instance_metrics(gt, pred, iou_threshold=0.5):
    """
    Computes instance-level metrics for instance segmentation using a vectorized approach.

    Parameters:
      gt (ndarray): Ground truth label image with instances labeled as nonzero integers.
      pred (ndarray): Predicted label image with instances labeled as nonzero integers.
      iou_threshold (float): IoU threshold to consider a pair as a valid match.

    Returns:
      dict: Dictionary with keys:
        - 'TP': Number of true positives.
        - 'FP': Number of false positives.
        - 'FN': Number of false negatives.
        - 'precision': Precision score.
        - 'recall': Recall score.
        - 'f1': F1 score.
        - 'mean_instance_iou': Mean IoU over matched instances.
        - 'mean_instance_dice': Mean Dice coefficient over matched instances.
    """

    # Get unique instance labels (exclude background = 0)
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]

    # Handle edge cases
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return {
            'TP': 0, 'FP': 0, 'FN': 0,
            'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
            'mean_instance_iou': np.nan, 'mean_instance_dice': np.nan
        }
    if len(gt_ids) == 0:
        return {
            'TP': 0, 'FP': len(pred_ids), 'FN': 0,
            'precision': 0.0, 'recall': np.nan, 'f1': np.nan,
            'mean_instance_iou': np.nan, 'mean_instance_dice': np.nan
        }
    if len(pred_ids) == 0:
        return {
            'TP': 0, 'FP': 0, 'FN': len(gt_ids),
            'precision': np.nan, 'recall': 0.0, 'f1': np.nan,
            'mean_instance_iou': np.nan, 'mean_instance_dice': np.nan
        }

    # Build confusion matrix using histogram2d.
    # Bins are set such that each label (including 0) gets its own bin.
    bins_gt = np.arange(0, gt.max() + 2)
    bins_pred = np.arange(0, pred.max() + 2)
    conf_matrix, _, _ = np.histogram2d(gt.ravel(), pred.ravel(), bins=(bins_gt, bins_pred))
    # Exclude background (first row and column)
    intersect_matrix = conf_matrix[1:, 1:]  # rows correspond to GT instances, columns to predicted instances

    # Compute area (pixel counts) per instance from ground truth and predicted images.
    gt_area = np.bincount(gt.ravel())[1:]
    pred_area = np.bincount(pred.ravel())[1:]

    # Compute union for each GT-pred pair: union = area(gt) + area(pred) - intersection.
    union_matrix = gt_area[:, None] + pred_area[None, :] - intersect_matrix
    # Compute IoU and Dice matrices
    iou_matrix = np.where(union_matrix > 0, intersect_matrix / union_matrix, 0)
    dice_matrix = np.where(
        (gt_area[:, None] + pred_area[None, :]) > 0,
        2 * intersect_matrix / (gt_area[:, None] + pred_area[None, :]),
        0
    )

    # Use the Hungarian algorithm to obtain one-to-one matching that maximizes the IoU.
    # linear_sum_assignment minimizes cost so we use negative IoU.
    cost_matrix = -iou_matrix
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Filter matched pairs by the IoU threshold.
    matched_iou = iou_matrix[gt_indices, pred_indices]
    valid_matches = matched_iou >= iou_threshold
    final_gt_indices = gt_indices[valid_matches]
    final_pred_indices = pred_indices[valid_matches]
    final_iou = matched_iou[valid_matches]
    final_dice = dice_matrix[gt_indices, pred_indices][valid_matches]

    TP = len(final_iou)
    FN = len(gt_ids) - TP
    FP = len(pred_ids) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    mean_instance_iou = np.mean(final_iou) if TP > 0 else np.nan
    mean_instance_dice = np.mean(final_dice) if TP > 0 else np.nan

    metrics = {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_instance_iou': mean_instance_iou,
        'mean_instance_dice': mean_instance_dice
    }
    return metrics


def test_model_performance_widget():

    @magicgui(
        iou_threshold=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1, label='IoU Threshold'),
        call_button='Calculate Performance',
        layout='vertical',
    )
    def inner_widget(
            prediction_label_image: Labels,
            ground_truth_label_image: Labels,
            iou_threshold: float
    ):
        """
        Computes and displays pixel-level and instance-level accuracy metrics between two label images.
        """
        # For pixel-level metrics, the images are binarized inside the function.
        pred = prediction_label_image.data
        gt = ground_truth_label_image.data

        overall_acc, per_label_acc, mean_iou, mean_dice = compute_pixel_metrics(pred, gt)

        print("=== Pixel-Level Metrics ===")
        print("Overall Accuracy: {:.4f}".format(overall_acc))
        print("Per-Label Accuracy:")
        for label, acc in per_label_acc.items():
            print(f"  Label {label}: {acc:.4f}")
        print("Mean IoU (Pixel-Level): {:.4f}".format(mean_iou))
        print("Mean Dice (Pixel-Level): {:.4f}".format(mean_dice))
        print()

        # Compute instance-level metrics (using the original non-binarized instance labeling).
        instance_metrics = compute_instance_metrics(gt, pred, iou_threshold=iou_threshold)

        print("=== Instance-Level Metrics ===")
        print("True Positives (TP):", instance_metrics['TP'])
        print("False Positives (FP):", instance_metrics['FP'])
        print("False Negatives (FN):", instance_metrics['FN'])
        print("Precision: {:.4f}".format(instance_metrics['precision']))
        print("Recall: {:.4f}".format(instance_metrics['recall']))
        print("F1 Score: {:.4f}".format(instance_metrics['f1']))
        print("Mean Instance IoU: {:.4f}".format(instance_metrics['mean_instance_iou']))
        print("Mean Instance Dice: {:.4f}".format(instance_metrics['mean_instance_dice']))

    return inner_widget
