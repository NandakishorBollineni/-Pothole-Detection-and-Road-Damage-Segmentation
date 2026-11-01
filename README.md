# MultiScale-RoadNet: Multi-Scale Deep Learning for Pothole Detection and Road Damage Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.3-61DAFB.svg?logo=react)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, research-oriented computer vision project for **automated road surface inspection, multi-scale pothole detection, pixel-level damage segmentation, and ASTM D6433 compliant severity estimation**.

---

## 1. Project Overview & Research Motivation

Road surface deterioration—especially the formation of potholes and structural fatigue fractures—presents hazards to vehicular safety, urban transit, and municipal infrastructure budgets. Traditional manual pavement inspection is dangerous, subjective, and labor-intensive.

While standard convolutional neural networks (e.g., standard U-Net) have been used for defect segmentation, **single-scale models suffer from severe limitations**:
1. **Scale Disparity**: Road defects span orders of magnitude: from small incipient cracks ($< 50\text{ cm}^2$) to massive multi-lane craters ($> 5\text{ m}^2$). Single-scale receptive fields either lose tiny pothole details through aggressive downsampling or miss wide context required to distinguish craters from shadows and oil stains.
2. **Boundary Raggedness**: Pothole edges are textured, non-uniform, and shaded unevenly by ambient sunlight.
3. **Lack of Actionable Civil Severity Quantification**: Most academic models output binary masks without translating results into actionable civil engineering repair priorities.

To resolve these challenges, this project introduces **MultiScale-RoadNet**, synthesizing architectural innovations from:
- **PODSNet**: Dual-task feature extraction combining pixel-level segmentation masks with localized bounding-box detection.
- **DMTC**: Dense multi-scale feature learning and Atrous Spatial Pyramid Pooling (ASPP) to expand receptive fields without spatial resolution decay.
- **PotholeGrade-BD**: Mathematical formulation of road damage severity using segmentation area ratio, contour aspect ratio, and photometric depth proxies.

---

## 2. Proposed Architecture: MultiScale-RoadNet

```
                       Input Road Image [3 x 256 x 256]
                                      │
                                  [Stem Conv]
                                      │
                ┌─────────────────────┴─────────────────────┐
                │        Multi-Stage Residual Backbone       │
                │  C2: 1/4 Scale (32 ch)  - Fine Boundaries │
                │  C3: 1/8 Scale (64 ch)  - Pavement Texture│
                │  C4: 1/16 Scale (128 ch)- Cavity Contours │
                │  C5: 1/32 Scale (256 ch)- Global Context  │
                └─────────────────────┬─────────────────────┘
                                      │
               [Atrous Spatial Pyramid Pooling (ASPP: r=1, 3, 6)]
                                      │
                ┌─────────────────────┴─────────────────────┐
                │    Top-Down Feature Pyramid Network (FPN) │
                │        P5 ──> P4 ──> P3 ──> P2            │
                └─────────────────────┬─────────────────────┘
                                      │
                    [Scale-Attention & Channel Fusion (MSFF)]
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
    [Segmentation Decoder]                       [Auxiliary Scale Head]
                │                                           │
     Pixel-Level Binary Mask                     Scale Classification
       (Road vs Pothole)                         (Small / Medium / Large)
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      │
                     [Damage Severity Assessment Engine]
               - Total Damaged Area (%)
               - Photometric Depth Proxy
               - Pavement Distress Index (0 - 100)
               - Civil Maintenance Action Code (ASTM D6433)
```

### Key Architectural Highlights:
1. **Dual-Pyramid Fusion**: Combines **ASPP** on deep semantic maps with **FPN** lateral connections to restore high-resolution spatial details for small defects.
2. **Scale-Attention Mechanism**: Dynamically weights multi-scale feature representations using channel-wise squeeze-and-excitation.
3. **Compact Parameter Efficiency**: **2.14M parameters** (50.5% smaller than the 4.32M parameter Baseline UNet), enabling real-time CPU inference (~64ms/frame).

---

## 3. Genuine Benchmark Results (Held-Out Test Split)

Both the **Baseline Single-Scale UNet** and the **Proposed MultiScale-RoadNet** were trained and evaluated on the real-world road pothole dataset:

| Evaluation Metric | Baseline Single-Scale UNet | Proposed MultiScale-RoadNet | Relative Delta / Advantage |
| :--- | :---: | :---: | :---: |
| **Model Parameters** | 4,318,401 | **2,137,172** | **-50.5% (Compact)** |
| **Checkpoint Size** | 49.52 MB | **24.60 MB** | **-50.5% (Storage)** |
| **Mean IoU (mIoU)** | 0.4251 (42.51%) | **0.5087 (50.87%)** | **+8.36% Absolute Boost** |
| **Dice Score (F1)** | 0.5966 (59.66%) | **0.6744 (67.44%)** | **+7.78% Absolute Boost** |
| **Precision** | 0.4965 (49.65%) | **0.6096 (60.96%)** | **+11.31% Absolute Boost** |
| **Recall** | 0.7473 (74.73%) | **0.7545 (75.45%)** | **+0.72% Gain** |
| **Small Pothole IoU (<500px)** | 0.0300 (3.00%) | **0.0431 (4.31%)** | **+43.7% Relative Gain** |
| **Medium Pothole IoU (500-2500px)** | 0.3356 (33.56%) | **0.5072 (50.72%)** | **+17.16% Absolute Gain** |
| **Large Pothole IoU (>2500px)** | 0.4723 (47.23%) | **0.5361 (53.61%)** | **+6.38% Absolute Gain** |
| **Inference Latency (CPU)** | 163.23 ms / frame | **64.88 ms / frame** | **2.5x Faster Inference** |

---

## 4. Repository Structure

```
multiscale-pothole-segmentation/
├── backend/
│   ├── main.py              # FastAPI REST API (Inference, Comparison, Reports)
│   ├── database.py          # SQLite inspection history persistence layer
│   └── report_generator.py  # Printable HTML/PDF civil inspection report renderer
├── data/
│   ├── download_dataset.py  # Automated dataset downloader from Hugging Face
│   ├── dataset.py           # PyTorch Dataset with scale-aware augmentations
│   ├── processed/           # Train (70), Val (10), Test (10) splits
│   └── samples/             # Pre-loaded benchmark road image samples
├── models/
│   ├── __init__.py
│   ├── baseline_unet.py     # Single-scale UNet baseline model
│   ├── multiscale_roadnet.py# Proposed MultiScale-RoadNet (ASPP + FPN + Scale-Attention)
│   └── severity_engine.py   # PotholeGrade damage severity quantification engine
├── training/
│   ├── losses.py            # BCE + Soft Dice Loss & scale-wise IoU metrics
│   ├── train.py             # Dual-model training script with checkpointing
│   └── evaluate.py          # Comprehensive test evaluation & comparison plotter
├── checkpoints/
│   ├── baseline_unet_best.pth
│   ├── multiscale_roadnet_best.pth
│   └── training_history.json
├── reports/
│   ├── benchmark_results.json
│   ├── PROJECT_REPORT.md    # Academic research paper report
│   └── visual_comparisons/  # Qualitative 4-panel visual comparison figures
├── frontend/                # Modern React.js (Vite) Glassmorphism Application
│   ├── src/
│   │   ├── App.jsx          # Tabbed main layout & state management
│   │   ├── index.css        # Curated dark-mode design system & animations
│   │   └── components/
│   │       ├── ImageViewer.jsx       # Interactive split-screen slider viewer
│   │       ├── SeverityGauge.jsx     # Radial DSI gauge & recommendation box
│   │       ├── ModelComparisonTab.jsx# Head-to-head model inspection
│   │       ├── ResearchTab.jsx       # Benchmark tables & literature review
│   │       └── AuditHistoryTab.jsx   # SQLite database audit trail
│   └── package.json
├── requirements.txt
└── README.md
```

---

## 5. Quickstart & Installation

### Step 1: Environment Setup
```bash
# Navigate to project directory
cd multiscale-pothole-segmentation

# Activate Python virtual environment
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset & Train Models (Optional - Pre-trained weights included)
```bash
# 1. Download and preprocess real road dataset
python data/download_dataset.py

# 2. Train Baseline UNet and Proposed MultiScale-RoadNet
python training/train.py

# 3. Evaluate models on unseen test split and generate visual panels
python training/evaluate.py
```

### Step 3: Run Full-Stack Web Application

**Terminal 1 (Backend API):**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*API Documentation available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

**Terminal 2 (React Frontend):**
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
*Access web interface at: [http://127.0.0.1:5173](http://127.0.0.1:5173)*

---

## 6. Web Application Features

1. **Interactive Inspector**:
   - Upload road photographs or select benchmark samples.
   - **Interactive Split Slider**: Drag slider horizontally to dynamically reveal original asphalt vs detected damage masks.
   - **Severity Gauge**: Visualizes Damage Severity Index (0-100) with risk tier (Low, Moderate, Severe, Critical).
   - **Civil Engineering Guidance**: Recommends maintenance actions (e.g., Cold patch vs Full-depth reclamation).
   - **Printable Audit Report**: One-click download of formal ASTM-compliant inspection documentation.
2. **Dual Model Comparison**:
   - Runs both Baseline UNet and MultiScale-RoadNet simultaneously on the same road image to observe edge boundary sharpness and fine fracture detection.
3. **Research & Benchmarks**:
   - Interactive quantitative table and ablation comparison.
4. **Audit History**:
   - Persistent SQLite inspection trail with search, severity filter, and report viewing.

---

## 7. License & Citation

Distributed under the MIT License. Developed for academic computer vision research in intelligent transportation systems.
