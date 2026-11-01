"""
Loss Functions and Quantitative Evaluation Metrics for Road Damage Segmentation.
Implements BCE + Soft Dice Loss and Scale-Aware Evaluation Metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Tuple


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Mitigates class imbalance between road surface and damaged pothole regions.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class CombinedBCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy (BCE) and Soft Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Computes IoU, Dice Score (F1), Precision, and Recall on batch tensors.
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_pred = preds_flat.sum().item()
    total_target = targets_flat.sum().item()
    union = total_pred + total_target - intersection

    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
    precision = (intersection + smooth) / (total_pred + smooth)
    recall = (intersection + smooth) / (total_target + smooth)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def calculate_scale_wise_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    small_thresh: int = 500,
    med_thresh: int = 2500,
) -> Dict[str, float]:
    """
    Computes scale-disaggregated IoU to evaluate performance on Small, Medium, and Large potholes.
    Demonstrates the quantitative advantage of MultiScale-RoadNet over single-scale baseline.
    """
    pred_bin = (pred_mask > 0.5).astype(np.uint8)
    target_bin = (target_mask > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_metrics = {
        "small_iou": [],
        "medium_iou": [],
        "large_iou": [],
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        instance_mask = np.zeros_like(target_bin)
        cv2.drawContours(instance_mask, [cnt], -1, 1, -1)

        # Intersect with prediction
        inter = np.logical_and(instance_mask, pred_bin).sum()
        union = np.logical_or(instance_mask, pred_bin).sum()
        iou = float(inter + 1e-6) / float(union + 1e-6)

        if area < small_thresh:
            scale_metrics["small_iou"].append(iou)
        elif area <= med_thresh:
            scale_metrics["medium_iou"].append(iou)
        else:
            scale_metrics["large_iou"].append(iou)

    return {
        "small_iou": float(np.mean(scale_metrics["small_iou"])) if scale_metrics["small_iou"] else 0.0,
        "medium_iou": float(np.mean(scale_metrics["medium_iou"])) if scale_metrics["medium_iou"] else 0.0,
        "large_iou": float(np.mean(scale_metrics["large_iou"])) if scale_metrics["large_iou"] else 0.0,
    }
