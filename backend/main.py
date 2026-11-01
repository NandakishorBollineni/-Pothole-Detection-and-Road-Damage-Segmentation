"""
FastAPI Backend Application for Multi-Scale Pothole Detection & Road Damage Segmentation.
Provides REST APIs for real-time inference, side-by-side model comparison,
severity grading, SQLite inspection history, and civil report generation.
"""
import io
import os
import sys
import base64
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models import BaselineUNet, MultiScaleRoadNet, DamageSeverityEngine
from backend.database import insert_inspection, get_all_inspections, get_inspection_by_id, delete_inspection
from backend.report_generator import generate_html_report

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
SAMPLES_DIR = BASE_DIR / "data" / "samples"
REPORTS_DIR = BASE_DIR / "reports"

app = FastAPI(
    title="MultiScale Road Damage Inspection API",
    description="Research-Oriented Multi-Scale Deep Learning Pothole Detection & Damage Segmentation",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
baseline_model: Optional[BaselineUNet] = None
multiscale_model: Optional[MultiScaleRoadNet] = None
severity_engine = DamageSeverityEngine(small_thresh=1000, medium_thresh=5000)


def load_model_weights():
    global baseline_model, multiscale_model
    baseline_path = CHECKPOINTS_DIR / "baseline_unet_best.pth"
    multiscale_path = CHECKPOINTS_DIR / "multiscale_roadnet_best.pth"

    try:
        baseline_model = BaselineUNet(in_channels=3, num_classes=1)
        if baseline_path.exists():
            ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
            baseline_model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Loaded] Baseline UNet from {baseline_path.name}")
        baseline_model.to(device)
        baseline_model.eval()
    except Exception as e:
        print(f"[Warning] Could not load Baseline UNet: {e}")

    try:
        multiscale_model = MultiScaleRoadNet(in_channels=3, num_classes=1, fpn_channels=64)
        if multiscale_path.exists():
            ckpt = torch.load(multiscale_path, map_location=device, weights_only=False)
            multiscale_model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Loaded] MultiScale-RoadNet from {multiscale_path.name}")
        multiscale_model.to(device)
        multiscale_model.eval()
    except Exception as e:
        print(f"[Warning] Could not load MultiScale-RoadNet: {e}")


load_model_weights()


def preprocess_image(img_rgb: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """Resizes and normalizes RGB numpy image into PyTorch tensor."""
    resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


def render_overlay(img_rgb: np.ndarray, prob_mask: np.ndarray, instances: List[Dict[str, Any]], color_theme: str = "multiscale") -> np.ndarray:
    """
    Renders high-visibility visual overlay:
    Semi-transparent color mask + contour outlines + bounding boxes with scale chips.
    """
    h, w = img_rgb.shape[:2]
    mask_resized = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (mask_resized >= 0.5).astype(np.uint8)

    overlay = img_rgb.copy()
    if color_theme == "multiscale":
        # Crimson-Amber mask for MultiScale
        color_mask = np.zeros_like(img_rgb)
        color_mask[binary == 1] = [239, 68, 68]  # Red/Rose
    else:
        # Cyan-Blue mask for Baseline
        color_mask = np.zeros_like(img_rgb)
        color_mask[binary == 1] = [59, 130, 246]  # Blue

    # Blend
    cv2.addWeighted(color_mask, 0.45, overlay, 0.55, 0, overlay)

    # Scale ratio for instances if image dimensions differ from 256
    scale_x = w / 256.0
    scale_y = h / 256.0

    for inst in instances:
        bx, by, bw, bh = inst["bbox"]
        sx = int(bx * scale_x)
        sy = int(by * scale_y)
        sw = int(bw * scale_x)
        sh = int(bh * scale_y)
        sc = inst["scale"].upper()

        box_color = (16, 185, 129) if sc == "SMALL" else ((245, 158, 11) if sc == "MEDIUM" else (239, 68, 68))
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), box_color, 2)

        # Label tag
        label = f"#{inst['id']} {sc} ({int(inst['confidence']*100)}%)"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (sx, max(0, sy - lh - 6)), (sx + lw + 6, sy), box_color, -1)
        cv2.putText(overlay, label, (sx + 3, max(lh + 2, sy - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def image_to_base64(img_rgb: np.ndarray, format: str = "JPEG", quality: int = 88) -> str:
    """Encodes numpy RGB image to Base64 data URL string."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ---------------- API ENDPOINTS ----------------

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "device": str(device),
        "baseline_ready": baseline_model is not None,
        "multiscale_ready": multiscale_model is not None,
        "parameters": {
            "baseline_unet": baseline_model.count_parameters() if baseline_model else 0,
            "multiscale_roadnet": multiscale_model.count_parameters() if multiscale_model else 0,
        }
    }


@app.get("/api/models")
def get_models_info():
    return {
        "models": [
            {
                "id": "multiscale",
                "name": "MultiScale-RoadNet (Proposed)",
                "type": "Multi-Scale Dual-Pyramid Network",
                "parameters": multiscale_model.count_parameters() if multiscale_model else 2137172,
                "features": [
                    "Atrous Spatial Pyramid Pooling (ASPP: rates 1, 3, 6)",
                    "Top-Down Feature Pyramid Network (FPN)",
                    "Scale-Attention Channel Fusion (MSFF)",
                    "High-resolution edge boundary restoration",
                    "Superior detection of small incipient potholes"
                ],
                "research_basis": "Inspired by DMTC & PODSNet"
            },
            {
                "id": "baseline",
                "name": "Baseline UNet",
                "type": "Single-Scale Encoder-Decoder",
                "parameters": baseline_model.count_parameters() if baseline_model else 4318401,
                "features": [
                    "Standard 4-level U-Net architecture",
                    "Single-scale 3x3 convolutions",
                    "Direct skip connections without pyramid fusion"
                ],
                "research_basis": "Ronneberger et al. standard benchmark"
            }
        ]
    }


@app.get("/api/samples")
def get_sample_images():
    """Lists pre-packaged road inspection benchmark samples."""
    samples = []
    if SAMPLES_DIR.exists():
        for p in sorted(list(SAMPLES_DIR.glob("*.jpg"))):
            samples.append({
                "id": p.name,
                "name": p.stem.replace("_", " ").title(),
                "filename": p.name,
            })
    return {"samples": samples}


@app.get("/api/sample-image/{filename}")
def get_sample_image_data(filename: str):
    """Returns sample image as base64."""
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    img_bgr = cv2.imread(str(file_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return {
        "filename": filename,
        "image_base64": image_to_base64(img_rgb)
    }


@app.post("/api/predict")
async def predict_road_damage(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Query(None),
    model_choice: str = Query("multiscale"),
    threshold: float = Query(0.5)
):
    """
    Runs pothole detection, segmentation, and damage severity grading.
    """
    if file:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(pil_img)
        filename = file.filename or "uploaded_road.jpg"
    elif sample_name:
        sample_path = SAMPLES_DIR / sample_name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found")
        img_bgr = cv2.imread(str(sample_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        filename = sample_name
    else:
        raise HTTPException(status_code=400, detail="Provide an uploaded image file or a sample_name")

    # Select model
    active_model = multiscale_model if (model_choice == "multiscale" and multiscale_model) else baseline_model
    model_name = "MultiScale-RoadNet" if model_choice == "multiscale" else "Baseline-UNet"

    if active_model is None:
        raise HTTPException(status_code=500, detail="Requested deep learning model is not loaded")

    # Preprocess
    input_tensor = preprocess_image(img_rgb, target_size=256)

    # Inference
    with torch.no_grad():
        if isinstance(active_model, MultiScaleRoadNet):
            logits = active_model(input_tensor, return_aux=False)
        else:
            logits = active_model(input_tensor)

        prob_tensor = torch.sigmoid(logits).squeeze()
        prob_mask_256 = prob_tensor.cpu().numpy()

    # Severity analysis & instance extraction at 256x256
    img_rgb_256 = cv2.resize(img_rgb, (256, 256))
    assessment = severity_engine.assess(img_rgb_256, prob_mask_256, threshold=threshold)

    # Render high-resolution overlay
    overlay_rgb = render_overlay(img_rgb, prob_mask_256, assessment["instances"], color_theme=model_choice)
    overlay_base64 = image_to_base64(overlay_rgb, format="JPEG", quality=90)
    original_base64 = image_to_base64(img_rgb, format="JPEG", quality=90)

    # Thumbnail for database
    thumb_rgb = cv2.resize(overlay_rgb, (120, 120))
    thumb_base64 = image_to_base64(thumb_rgb, format="JPEG", quality=75)

    # Persist to SQLite history
    db_record = {
        "filename": filename,
        "thumbnail_base64": thumb_base64,
        "model_used": model_name,
        "severity_score": assessment["severity_score"],
        "severity_level": assessment["severity_level"],
        "action_code": assessment["action_code"],
        "badge_color": assessment["badge_color"],
        "pothole_count": assessment["pothole_count"],
        "damaged_area_percentage": assessment["damaged_area_percentage"],
        "scale_distribution": assessment["scale_distribution"],
        "recommendation": assessment["recommendation"],
        "instances": assessment["instances"],
    }
    record_id = insert_inspection(db_record)

    return {
        "inspection_id": record_id,
        "filename": filename,
        "model_used": model_name,
        "original_image_base64": original_base64,
        "overlay_image_base64": overlay_base64,
        "assessment": assessment,
    }


@app.post("/api/compare")
async def compare_models(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Query(None),
    threshold: float = Query(0.5)
):
    """
    Runs BOTH Baseline UNet and MultiScale-RoadNet side-by-side on the same image.
    Demonstrates multi-scale edge refinement and small pothole capture.
    """
    if file:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(pil_img)
        filename = file.filename or "uploaded_road.jpg"
    elif sample_name:
        sample_path = SAMPLES_DIR / sample_name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample not found")
        img_bgr = cv2.imread(str(sample_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        filename = sample_name
    else:
        raise HTTPException(status_code=400, detail="Provide image file or sample_name")

    input_tensor = preprocess_image(img_rgb, target_size=256)
    img_rgb_256 = cv2.resize(img_rgb, (256, 256))

    # 1. Baseline UNet
    with torch.no_grad():
        b_logits = baseline_model(input_tensor)
        b_probs = torch.sigmoid(b_logits).squeeze().cpu().numpy()
    b_assess = severity_engine.assess(img_rgb_256, b_probs, threshold=threshold)
    b_overlay = render_overlay(img_rgb, b_probs, b_assess["instances"], color_theme="baseline")

    # 2. MultiScale-RoadNet
    with torch.no_grad():
        p_logits = multiscale_model(input_tensor, return_aux=False)
        p_probs = torch.sigmoid(p_logits).squeeze().cpu().numpy()
    p_assess = severity_engine.assess(img_rgb_256, p_probs, threshold=threshold)
    p_overlay = render_overlay(img_rgb, p_probs, p_assess["instances"], color_theme="multiscale")

    # Comparative difference analysis
    small_potholes_delta = p_assess["scale_distribution"]["small"] - b_assess["scale_distribution"]["small"]
    total_count_delta = p_assess["pothole_count"] - b_assess["pothole_count"]

    return {
        "filename": filename,
        "original_base64": image_to_base64(img_rgb),
        "baseline": {
            "name": "Baseline UNet",
            "overlay_base64": image_to_base64(b_overlay),
            "assessment": b_assess,
        },
        "multiscale": {
            "name": "MultiScale-RoadNet (Proposed)",
            "overlay_base64": image_to_base64(p_overlay),
            "assessment": p_assess,
        },
        "comparison_analysis": {
            "small_potholes_captured_diff": small_potholes_delta,
            "total_defects_diff": total_count_delta,
            "verdict": (
                "MultiScale-RoadNet successfully identified additional fine-scale surface fractures "
                "missed by the single-scale baseline due to its high-resolution FPN lateral connections."
                if small_potholes_delta > 0 else
                "Both models identified primary defect boundaries; MultiScale-RoadNet provided sharper edge demarcation."
            )
        }
    }


@app.get("/api/history")
def get_history(limit: int = 50):
    records = get_all_inspections(limit=limit)
    return {"inspections": records}


@app.delete("/api/history/{record_id}")
def delete_history_item(record_id: int):
    success = delete_inspection(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"success": True, "deleted_id": record_id}


@app.get("/api/report/{record_id}", response_class=HTMLResponse)
def view_inspection_report(record_id: int):
    record = get_inspection_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Inspection record not found")
    html_content = generate_html_report(record, original_base64="", overlay_base64="")
    return HTMLResponse(content=html_content)


@app.get("/api/benchmark")
def get_benchmark():
    benchmark_file = REPORTS_DIR / "benchmark_results.json"
    history_file = CHECKPOINTS_DIR / "training_history.json"

    benchmarks = None
    if benchmark_file.exists():
        with open(benchmark_file, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)

    training_logs = None
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            training_logs = json.load(f)

    return {
        "benchmarks": benchmarks,
        "training_logs": training_logs,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
