"""
Dataset Downloader and Preprocessing Pipeline.
Downloads real road pothole segmentation dataset from Keremberke / Roboflow,
parses COCO polygon annotations, renders pixel-precise binary ground truth masks,
and organizes data into train, val, test, and sample benchmark splits.
"""
import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "keremberke/pothole-segmentation"
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
SAMPLES_DIR = BASE_DIR / "samples"
RAW_DIR = BASE_DIR / "raw"


def extract_split(zip_filename: str, split_name: str) -> Dict[str, Any]:
    """
    Downloads, extracts, and parses a dataset split into images and ground-truth masks.
    """
    print(f"\n[Dataset] Processing split: {split_name} ({zip_filename})...")
    local_zip_path = hf_hub_download(repo_id=REPO_ID, filename=zip_filename, repo_type="dataset")

    # Temporary extraction folder
    temp_dir = RAW_DIR / split_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, "r") as z:
        z.extractall(temp_dir)

    coco_json_path = temp_dir / "_annotations.coco.json"
    if not coco_json_path.exists():
        print(f"[Warning] No _annotations.coco.json in {temp_dir}")
        return {"images": 0, "instances": 0}

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Destination directories
    img_dest_dir = PROCESSED_DIR / split_name / "images"
    mask_dest_dir = PROCESSED_DIR / split_name / "masks"
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    mask_dest_dir.mkdir(parents=True, exist_ok=True)

    # Map image_id to annotations
    img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        img_id_to_anns.setdefault(img_id, []).append(ann)

    processed_count = 0
    total_instances = 0
    scale_counts = {"small": 0, "medium": 0, "large": 0}

    for img_info in coco_data.get("images", []):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        src_img_path = temp_dir / file_name

        if not src_img_path.exists():
            continue

        # Load image
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Create blank mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw polygons for all pothole annotations on this image
        anns = img_id_to_anns.get(img_id, [])
        for ann in anns:
            segmentation = ann.get("segmentation", [])
            for seg in segmentation:
                if len(seg) >= 6:  # At least 3 points (x, y)
                    pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
                    total_instances += 1

                    area = float(ann.get("area", cv2.contourArea(pts)))
                    if area < 1000:
                        scale_counts["small"] += 1
                    elif area <= 5000:
                        scale_counts["medium"] += 1
                    else:
                        scale_counts["large"] += 1

        # Standardize filename (e.g. sample_001.jpg)
        base_name = Path(file_name).stem
        dest_img_path = img_dest_dir / f"{base_name}.jpg"
        dest_mask_path = mask_dest_dir / f"{base_name}.png"

        cv2.imwrite(str(dest_img_path), img)
        cv2.imwrite(str(dest_mask_path), mask)
        processed_count += 1

        # If test or val split, also save some to samples directory for web application
        if split_name in ["val", "test"] and processed_count <= 4:
            SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(dest_img_path), str(SAMPLES_DIR / f"{split_name}_{base_name}.jpg"))

    print(f"  Processed {processed_count} images with {total_instances} pothole instances.")
    print(f"  Scales: Small={scale_counts['small']}, Medium={scale_counts['medium']}, Large={scale_counts['large']}")

    return {
        "images": processed_count,
        "instances": total_instances,
        "scales": scale_counts,
    }


def main():
    print("=" * 60)
    print("MultiScale-RoadNet: Downloading & Preparing Real Road Damage Dataset")
    print(f"Source: Hugging Face ({REPO_ID})")
    print("=" * 60)

    splits = [
        ("data/train.zip", "train"),
        ("data/valid.zip", "val"),
        ("data/test.zip", "test"),
    ]

    summary = {}
    for zip_file, split_name in splits:
        summary[split_name] = extract_split(zip_file, split_name)

    stats_file = BASE_DIR / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[Dataset Preparation Complete]")
    print(f"Dataset statistics saved to {stats_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
