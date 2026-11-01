"""
PyTorch Dataset and Multi-Scale Augmentation Pipeline for Road Pothole Segmentation.
Includes scale-aware supervision targets and photometric augmentations.
"""
import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF


class PotholeDataset(Dataset):
    """
    Dataset class for road damage segmentation with multi-scale supervision.
    Loads paired road images and binary ground-truth masks.
    """

    def __init__(
        self,
        split_dir: Path,
        img_size: int = 256,
        is_train: bool = True,
        small_thresh: int = 1000,
        med_thresh: int = 5000,
    ):
        self.split_dir = Path(split_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.small_thresh = small_thresh
        self.med_thresh = med_thresh

        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"

        # List all image files
        self.image_files: List[Path] = []
        if self.images_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                self.image_files.extend(list(self.images_dir.glob(ext)))
        self.image_files.sort()

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def _apply_augmentations(
        self, img_pil, mask_pil
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize to standard input resolution
        img_pil = TF.resize(img_pil, [self.img_size, self.img_size], interpolation=transforms.InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, [self.img_size, self.img_size], interpolation=transforms.InterpolationMode.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                img_pil = TF.hflip(img_pil)
                mask_pil = TF.hflip(mask_pil)

            # Random vertical flip
            if random.random() > 0.5:
                img_pil = TF.vflip(img_pil)
                mask_pil = TF.vflip(mask_pil)

            # Random rotation (+/- 15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img_pil = TF.rotate(img_pil, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # Color jitter for simulating road lighting variations (shadows, overcast, glare)
            if random.random() > 0.3:
                color_jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2)
                img_pil = color_jitter(img_pil)

        # Convert to Tensor
        img_tensor = TF.to_tensor(img_pil)  # [3, H, W] in [0, 1]
        img_tensor = self.normalize(img_tensor)

        mask_np = np.array(mask_pil)
        # Binarize mask: > 127 is damaged pothole
        mask_bin = (mask_np > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        # Load image (RGB)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load mask
        if mask_path.exists():
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask_gray = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

        # PIL conversion for torchvision transforms
        from PIL import Image
        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(mask_gray)

        img_tensor, mask_tensor = self._apply_augmentations(img_pil, mask_pil)

        # Calculate scale-supervision target for multi-scale aux head
        mask_np = (mask_tensor.squeeze(0).numpy() > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_small = 0.0
        has_medium = 0.0
        has_large = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (self.small_thresh / (img_rgb.shape[0] * img_rgb.shape[1])) * (self.img_size ** 2):
                has_small = 1.0
            elif area <= (self.med_thresh / (img_rgb.shape[0] * img_rgb.shape[1])) * (self.img_size ** 2):
                has_medium = 1.0
            else:
                has_large = 1.0

        scale_target = torch.tensor([has_small, has_medium, has_large], dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "scale_target": scale_target,
            "name": img_path.name,
        }


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
    img_size: int = 256,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory helper to construct train, val, and test dataloaders."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "processed" / "train"
    val_dir = data_dir / "processed" / "val"
    test_dir = data_dir / "processed" / "test"

    train_ds = PotholeDataset(train_dir, img_size=img_size, is_train=True)
    val_ds = PotholeDataset(val_dir, img_size=img_size, is_train=False)
    test_ds = PotholeDataset(test_dir, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
