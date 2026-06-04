# PPE Safety Detection System

Real-time Personal Protective Equipment (PPE) violation detection for construction and industrial environments. Fine-tuned YOLOv8 object detection model with ByteTrack multi-object tracking, served through a FastAPI inference API, persisted to PostgreSQL, and visualized on a Next.js dashboard deployed to Vercel.

---

## Live Demo

| Service | URL |
|---|---|
| Dashboard | `https://dashboard-nine-ebon-70.vercel.app/` |
| API (Swagger UI) | ` https://safety-violation-detection-cv.onrender.com/docs` |

> The API runs on Render's free tier and cold-starts after 15 minutes of inactivity. Use the "Wake up API" button in the dashboard and allow 30 seconds before uploading a video.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Detection Classes](#detection-classes)
- [Model Performance](#model-performance)
- [Inference Benchmark](#inference-benchmark)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Development](#local-development)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Engineering Decisions](#engineering-decisions)
- [Known Limitations](#known-limitations)

---

## Overview

This system addresses a core problem in workplace safety: manual PPE compliance monitoring is intermittent, subjective, and labor-intensive. An automated system that processes video footage and logs violations with timestamps, duration, and spatial location creates an auditable safety record with no human bottleneck.

The pipeline works as follows:

1. A video clip is uploaded through the Next.js dashboard
2. The FastAPI backend processes every frame through an ONNX-optimized YOLOv8 model
3. ByteTrack assigns persistent IDs to people across frames
4. A violation is confirmed only when a person has been non-compliant for at least 3 consecutive seconds — filtering out single-frame detection artifacts
5. Confirmed violations are written to PostgreSQL with timestamp, violation type, duration, and bounding box centroid
6. The dashboard renders violation history, type breakdown charts, and a spatial heatmap of where violations occur most frequently

---

## System Architecture

```
Video Upload (Browser)
        |
        v
Next.js Dashboard (Vercel)
        |
        | HTTP POST /analyze-video
        v
FastAPI Backend (Render)
        |
        |-- ONNX Inference Engine
        |       |-- YOLOv8s fine-tuned on PPE dataset
        |       |-- ONNXRuntime (2x faster than PyTorch inference)
        |
        |-- Violation Timer Logic
        |       |-- ByteTrack persistent track IDs
        |       |-- 90-frame (3s at 30fps) confirmation threshold
        |
        v
PostgreSQL (Supabase)
        |
        v
GET /violations
        |
        v
Next.js Dashboard (Vercel)
        |-- Bar chart: violations by type
        |-- Scatter plot: violation location heatmap
        |-- Sortable violation log table
```

---

## Detection Classes

The model is trained on the [Construction Site Safety dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) (version 27, 2,588 training images, CC BY 4.0).

| Index | Class | Type |
|---|---|---|
| 0 | Hardhat | Compliant |
| 1 | Mask | Compliant |
| 2 | NO-Hardhat | **Violation** |
| 3 | NO-Mask | Violation |
| 4 | NO-Safety Vest | **Violation** |
| 5 | Person | Context |
| 6 | Safety Cone | Context |
| 7 | Safety Vest | Compliant |
| 8 | machinery | Context |
| 9 | vehicle | Context |

The system specifically monitors classes 2 (NO-Hardhat) and 4 (NO-Safety Vest) as actionable violations. The remaining classes provide contextual detection that improves overall model accuracy through richer scene understanding during training.

---

## Model Performance

Trained for 50 epochs on an NVIDIA RTX 4060 Ti (16GB). Training completed in 25 minutes.

**Test set results (held-out, not seen during training):**

| Metric | Value |
|---|---|
| mAP@50 (overall) | 0.847 |
| mAP@50-95 (overall) | 0.554 |
| Precision | 0.937 |
| Recall | 0.769 |

**Per-class AP@50 on test set:**

| Class | AP@50 |
|---|---|
| Hardhat | 0.852 |
| Mask | 0.981 |
| NO-Hardhat | 0.761 |
| NO-Mask | 0.798 |
| NO-Safety Vest | 0.810 |
| Person | 0.858 |
| Safety Cone | 0.877 |
| Safety Vest | 0.912 |
| machinery | 0.916 |
| vehicle | 0.702 |

The two primary violation classes (NO-Hardhat: 0.761, NO-Safety Vest: 0.810) both exceed 0.75 mAP@50, indicating reliable detection of the cases that matter most operationally.

---

## Inference Benchmark

Measured on NVIDIA RTX 4060 Ti, 100 runs after 5-run warmup, input size 640x640, batch size 1.

| Runtime | FPS | Latency (ms/frame) | Notes |
|---|---|---|---|
| PyTorch (.pt) | 18.4 | 54.3 | Baseline |
| ONNX Runtime | 36.1 | 27.7 | 1.96x speedup |

ONNX export was performed with `simplify=True` which runs the ONNX simplifier to fuse redundant operations before saving. Accuracy is identical to the PyTorch baseline — ONNX Runtime eliminates training infrastructure overhead (autograd, gradient tracking) that PyTorch carries even at inference time.

The ONNX model is what runs in the deployed Render backend. The `.pt` file is not present in the Docker container.

---

## Tech Stack

**Model training**
- YOLOv8s (Ultralytics 8.4.56)
- PyTorch 2.7.1 + CUDA 11.8
- ByteTrack (integrated into Ultralytics)

**Backend**
- FastAPI 0.111.0
- ONNX Runtime 1.18.0
- OpenCV 4.9.0 (headless build for Docker)
- SQLAlchemy 2.0.30
- PostgreSQL via Supabase (free tier)
- Docker + Uvicorn

**Frontend**
- Next.js (Pages Router)
- Chart.js + react-chartjs-2
- Deployed on Vercel

**Infrastructure**
- Render (backend, free tier, Docker runtime)
- Vercel (dashboard, free tier, no cold-start)
- Supabase (PostgreSQL, free tier, 500MB)
- GitHub (source, auto-deploy triggers)

---

## Project Structure

```
safety-detection-project/
|
|-- model/
|   |-- best.onnx                   # Exported YOLOv8s weights (23MB)
|
|-- backend/
|   |-- main.py                     # FastAPI routes
|   |-- inference.py                # ONNXEngine class: preprocess, run, postprocess
|   |-- database.py                 # SQLAlchemy models + Supabase connection
|   |-- requirements.txt            # Pinned production dependencies
|   |-- Dockerfile                  # python:3.10-slim + libgl1
|
|-- dashboard/
|   |-- pages/
|   |   |-- index.js                # Main dashboard: upload, charts, table
|   |-- package.json
|   |-- next.config.js
|   |-- .env.local                  # NEXT_PUBLIC_API_URL (not committed)
|
|-- notebooks/
|   |-- NB-01-Setup.ipynb           # Environment + dataset download
|   |-- NB-02-EDA.ipynb             # Class distribution, sample visualization
|   |-- NB-03-Training.ipynb        # Fine-tuning, evaluation, predictions
|   |-- NB-04-Tracking.ipynb        # ByteTrack + violation timer logic
|   |-- NB-05-Benchmark.ipynb       # ONNX export + FPS benchmarking
|
|-- docker-compose.yml              # Local dev: backend + postgres
|-- .gitignore
|-- README.md
```

---

## Local Development

### Prerequisites

- [Anaconda](https://www.anaconda.com) or Miniconda
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Node.js 18+](https://nodejs.org)
- NVIDIA GPU with CUDA 11.8 drivers (for training; inference runs on CPU too)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/safety-detection.git
cd safety-detection
```

### 2. Set up the Python environment (for notebooks)

```bash
conda create -n safety-cv python=3.10 -y
conda activate safety-cv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics onnxruntime opencv-python roboflow ipykernel pandas matplotlib
python -m ipykernel install --user --name safety-cv --display-name "Safety CV"
```

### 3. Download the dataset

```bash
# Option A — Roboflow Python API
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
project = rf.workspace('roboflow-universe-projects').project('construction-site-safety')
project.version(27).download('yolov8')
"

# Option B — Manual download from:
# https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
# Export format: YOLOv8
```

### 4. Run the notebooks in order

Open Jupyter in VS Code, select the "Safety CV" kernel, and run:

```
NB-01-Setup.ipynb       -- verify environment, inspect data.yaml
NB-02-EDA.ipynb         -- class distribution, bounding box visualization
NB-03-Training.ipynb    -- fine-tune YOLOv8s, evaluate on test set
NB-04-Tracking.ipynb    -- ByteTrack, violation logic
NB-05-Benchmark.ipynb   -- ONNX export, FPS benchmark
```

After NB-03, copy the best weights:

```bash
cp notebooks/runs/detect/runs/ppe/v1/weights/best.onnx model/best.onnx
```

### 5. Run the full stack locally

```bash
# Start backend + local postgres
docker-compose up --build

# Verify the API is running
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"loaded"}

# Start the dashboard (in a second terminal)
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

---

## Deployment

### Backend to Render

1. Push the repository to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repository
4. Set configuration:
   - Root Directory: `backend`
   - Runtime: `Docker`
   - Instance Type: `Free`
5. Add environment variable: `DATABASE_URL` = your Supabase connection URI
6. Deploy. First build takes 5-10 minutes.
7. Verify at `https://YOUR_SERVICE.onrender.com/docs`

### Dashboard to Vercel

**Option A — CLI**

```bash
npm install -g vercel
cd dashboard
vercel

# Set the production environment variable
vercel env add NEXT_PUBLIC_API_URL
# Value: https://YOUR_SERVICE.onrender.com

# Deploy to production
vercel --prod
```

**Option B — GitHub integration**

1. Go to [vercel.com](https://vercel.com) → Add New Project
2. Import your GitHub repository
3. Set Root Directory to `dashboard`
4. Add environment variable: `NEXT_PUBLIC_API_URL` = your Render URL
5. Deploy. Future pushes to `main` auto-deploy.

### Database (Supabase)

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to Settings → Database → Connection String (URI)
3. Use this URI as the `DATABASE_URL` environment variable in Render
4. The `violations` table is created automatically on first backend startup via `Base.metadata.create_all()`

---

## API Reference

Base URL: `https://YOUR_SERVICE.onrender.com`

Full interactive documentation available at `/docs` (Swagger UI).

### GET /health

Returns model load status. Use this to wake the Render instance before uploading.

```
Response 200:
{
  "status": "ok",
  "model": "loaded"
}
```

### POST /analyze-video

Upload a video file. The backend processes every frame, runs ONNX inference, applies violation timer logic, and writes confirmed violations to PostgreSQL.

```
Request:
  Content-Type: multipart/form-data
  Body: file (video/mp4, video/avi, video/mov)

Response 200:
{
  "frames_processed": 900,
  "violations_logged": 4
}
```

Processing time scales linearly with video length. A 30-second clip at 30fps (~900 frames) takes approximately 60-90 seconds on Render's free tier CPU.

### GET /violations

Returns logged violations in descending timestamp order.

```
Query parameters:
  limit  (integer, default 500)

Response 200:
[
  {
    "id": 1,
    "violation_type": "NO-Hardhat",
    "duration_seconds": 4.2,
    "timestamp": "2025-06-04T10:23:11.000Z",
    "bbox_x": 342.5,
    "bbox_y": 218.0
  }
]
```

`bbox_x` and `bbox_y` are the pixel coordinates of the bounding box centroid in the original frame. These are used by the dashboard to render the violation location scatter plot.

---

## Engineering Decisions

### Why ONNX Runtime instead of PyTorch at inference time

PyTorch is a training framework. At inference time it carries gradient tracking infrastructure, autograd state, and Python-level overhead that serves no purpose. ONNX Runtime is a dedicated inference engine that applies hardware-specific optimizations (operator fusion, memory layout optimization, SIMD vectorization) that PyTorch does not. The result on this hardware was a 1.96x speedup with zero change in output values. The `.pt` file is not present in the deployed Docker container — only `best.onnx`.

### Why the 3-second violation threshold

A single-frame detection of "NO-Hardhat" is not a reliable signal. Workers move quickly, partially occlude themselves, and the model has a non-zero false positive rate. A confirmed violation requires 90 consecutive frames (3 seconds at 30fps) of non-compliance from the same tracked individual. This threshold was chosen to match what a human safety observer would consider a genuine compliance failure vs a momentary detection artifact.

### Why ByteTrack over simpler tracking approaches

The naive alternative — detect violations per frame and count them — cannot measure duration because there is no concept of identity across frames. ByteTrack solves this by assigning persistent integer IDs to detected individuals across frames using a Kalman filter for motion prediction and IoU matching for association. ByteTrack's specific contribution over earlier trackers is that it does not discard low-confidence detections — it uses them in a second association pass to recover occluded tracks. This reduces ID switches, which is important because an ID switch resets the violation timer and would cause the system to under-report duration.

### Why Next.js instead of Streamlit

Streamlit is a Python library for building data science UIs quickly. It is appropriate for internal tools and notebooks but signals a data science prototype rather than a web application. Next.js produces a React application that deploys to Vercel's global CDN with no cold-start penalty on the free tier. The dashboard code is indistinguishable from a production frontend. Additionally, Vercel never spins down on the free tier — unlike Streamlit Community Cloud, which has its own availability constraints.

### Why not run ByteTrack in the deployed API

ByteTrack requires stateful tracking across frames — the tracker object must persist between frame calls. In the deployed FastAPI backend, the API is stateless by design: each `/analyze-video` call is independent. Running ByteTrack in the API would require either a persistent worker process or serializing/deserializing tracker state between requests, both of which add significant complexity. The current design processes the full video in a single request, which is appropriate for the portfolio use case. A production system would use a streaming architecture (e.g., a persistent worker consuming frames from a queue) rather than HTTP uploads.

---

## Known Limitations

**Render free tier cold-start** — The backend spins down after 15 minutes of inactivity. First request after sleep takes approximately 30 seconds. This is a platform constraint, not a bug. Click "Wake up API" in the dashboard and wait before uploading.

**No real-time streaming** — The system processes uploaded video files, not live RTSP streams. Real-time streaming would require a WebSocket or WebRTC transport layer and a persistent inference worker, which is outside the scope of this project.

**Single-threaded video processing** — The `/analyze-video` endpoint blocks until the entire video is processed. For long videos this can hit Render's request timeout. Keep test videos under 60 seconds.

**Track ID persistence across requests** — ByteTrack runs in the notebook environment for validation. The deployed API uses frame-level detection without persistent tracking (track_id is hardcoded to 0). Duration is still measured correctly because the violation timer runs within a single request, but a person who leaves and re-enters the frame within one video will have their timer reset.

**No authentication** — The API endpoints are publicly accessible. In a production environment, all routes should be protected by an authentication layer (API key or OAuth). The Supabase connection string is held server-side in Render environment variables and is never exposed to the browser.

---

## Dataset Citation

```
@misc{construction-site-safety_dataset,
  title        = {construction-site-safety Dataset},
  type         = {Open Source Dataset},
  author       = {Roboflow Universe Projects},
  howpublished = {Roboflow Universe},
  url          = {https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety},
  year         = {2024},
  note         = {visited on 2025-06-04},
}
```

License: CC BY 4.0

---

## License

MIT License. See `LICENSE` for details.
