"""
Model Training Pipeline.
Trains both Baseline UNet and Proposed MultiScale-RoadNet on the real road damage dataset.
Logs genuine epoch-by-epoch metrics and saves model checkpoints.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models import BaselineUNet, MultiScaleRoadNet
from data.dataset import get_dataloaders
from training.losses import CombinedBCEDiceLoss, calculate_metrics

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Trains a model for the specified number of epochs, logging genuine metrics.
    """
    print(f"\n{'='*60}")
    print(f"Starting Training: {model_name}")
    print(f"Architecture Parameters: {model.count_parameters():,}")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr}")
    print(f"{'='*60}")

    model.to(device)
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    aux_criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_precision": [],
        "val_recall": [],
        "lr": [],
    }

    best_val_iou = 0.0
    best_weights_path = CHECKPOINTS_DIR / f"{model_name.lower().replace('-', '_')}_best.pth"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            scale_targets = batch["scale_target"].to(device)

            optimizer.zero_grad()

            if isinstance(model, MultiScaleRoadNet):
                logits, aux_scales = model(images, return_aux=True)
                seg_loss = criterion(logits, masks)
                aux_loss = aux_criterion(aux_scales, scale_targets)
                loss = seg_loss + 0.2 * aux_loss
            else:
                logits = model(images)
                loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = running_train_loss / max(1, train_batches)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        all_logits = []
        all_masks = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                if isinstance(model, MultiScaleRoadNet):
                    logits = model(images, return_aux=False)
                else:
                    logits = model(images)

                loss = criterion(logits, masks)
                running_val_loss += loss.item()
                val_batches += 1

                all_logits.append(logits.cpu())
                all_masks.append(masks.cpu())

        avg_val_loss = running_val_loss / max(1, val_batches)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_masks = torch.cat(all_masks, dim=0)
        metrics = calculate_metrics(cat_logits, cat_masks)

        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_iou"].append(round(metrics["iou"], 4))
        history["val_dice"].append(round(metrics["dice"], 4))
        history["val_precision"].append(round(metrics["precision"], 4))
        history["val_recall"].append(round(metrics["recall"], 4))
        history["lr"].append(round(current_lr, 6))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val IoU: {metrics['iou']:.4f} | "
            f"Val Dice: {metrics['dice']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

        # Checkpoint if best IoU achieved
        if metrics["iou"] > best_val_iou:
            best_val_iou = metrics["iou"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": best_val_iou,
                "val_dice": metrics["dice"],
                "model_name": model_name,
            }, best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path.name} (IoU: {best_val_iou:.4f})")

    elapsed_time = time.time() - start_time
    print(f"\nFinished {model_name} in {elapsed_time:.1f}s. Best Val IoU: {best_val_iou:.4f}")

    return {
        "model_name": model_name,
        "parameters": model.count_parameters(),
        "best_val_iou": round(best_val_iou, 4),
        "best_weights_path": str(best_weights_path),
        "training_time_sec": round(elapsed_time, 1),
        "history": history,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Environment: Using {device}")

    # Load dataloaders
    data_dir = BASE_DIR / "data"
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        img_size=256,
        num_workers=0
    )

    epochs = 15

    # 1. Train Baseline UNet
    baseline_model = BaselineUNet(in_channels=3, num_classes=1)
    baseline_results = train_single_model(
        model_name="Baseline-UNet",
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )

    # 2. Train Proposed MultiScale-RoadNet
    proposed_model = MultiScaleRoadNet(in_channels=3, num_classes=1, fpn_channels=64)
    proposed_results = train_single_model(
        model_name="MultiScale-RoadNet",
        model=proposed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )

    # Save overall history
    all_history = {
        "baseline": baseline_results,
        "proposed": proposed_results,
        "epochs": epochs,
        "device": str(device),
        "dataset_size": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
    }

    history_path = CHECKPOINTS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"All training completed! History saved to {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
