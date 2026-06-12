# MRNet Volumetric MRI Classification API
MRNet-MLOps — Knee MRI Pathology Detection API An end-to-end MLOps pipeline that serves a multi-stream 3D CNN for knee MRI classification via a production-ready FastAPI service, containerized with Docker, monitored with Prometheus, and deployed on Hugging Face Spaces.

## Project Overview
This project takes a trained 3D CNN model (from the MRNet capstone project) and wraps it in a full MLOps pipeline:

- Gatekeeper CNN validates that each uploaded MRI file is the correct anatomical plane (axial/coronal/sagittal) before inference
- Accepts three MRI planes as .npy file uploads and runs multi-label diagnosis
- Returns predictions for ACL tear, Meniscus tear, and Abnormality with a clinical summary
- Uses Youden's J statistic for optimal classification thresholds
- Tracks inference metrics with Prometheus
- Fully containerized with Docker and deployed to the cloud

## Model Performance

### Main Diagnosis Model (3D CNN)

| Pathology | AUC   | Threshold (Youden's J) |
|-----------|-------|------------------------|
| ACL Tear  | 0.827 | 0.356                  |
| Meniscus  | 0.746 | 0.400                  |
| Abnormal  | 0.833 | 0.497                  |

### Gatekeeper Model (Plane Classifier)

| Plane     | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Axial     | 1.00      | 0.95   | 0.98 |
| Coronal   | 1.00      | 1.00   | 1.00 |
| Sagittal  | 0.98      | 1.00   | 0.99 |

**Overall Gatekeeper Accuracy: 99.1%**


## Repository Structure

```
MRNet-MLOps/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # CI/CD pipeline (GitHub Actions)
│
├── training/
│   └── train_gatekeeper.py         # Gatekeeper training script (real MRNet data)
│
├── app.py                          # FastAPI app with Gatekeeper + Prometheus metrics
├── mrnet_architecture.py           # Main 3D CNN model architecture & preprocessing
├── mrnet_3dcnn_artifacts.pth       # Trained diagnosis model weights (Git LFS)
├── gatekeeper_weights.pth          # Trained gatekeeper model weights (Git LFS)
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── test_api.py                     # API endpoint tests
├── .gitattributes                  # Git LFS tracking config
├── .gitignore                      # Ignored files
└── README.md                       # Project documentation
```

## Getting Started

### Option 1 — Use the Live API
 [https://sabnab-mrnet-api.hf.space/docs](https://sabnab-mrnet-api.hf.space/docs)

### Option 2 — Run Locally with Docker

```bash
# 1. Clone the repo
git clone https://github.com/bSabna/MRNet-MLOps.git
cd MRNet-MLOps

# 2. Build the Docker image
docker build -t mrnet-api .

# 3. Run the container
docker run -p 7860:7860 mrnet-api

# 4. Open the interactive API docs
# http://localhost:7860/docs
```

### Option 3 — Run Locally without Docker

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

##  API Endpoints

| Method | Endpoint   | Description                                |
|--------|------------|--------------------------------------------|
| GET    | `/`        | Root — health check                        |
| GET    | `/health`  | Returns `{"status": "healthy"}`            |
| POST   | `/predict` | Run gatekeeper validation + full diagnosis |
| GET    | `/metrics` | Prometheus metrics endpoint                |
| GET    | `/docs`    | Interactive Swagger UI                     |

### Example `/predict` Request

```python
import requests

files = {
    "sagittal": ("sagittal.npy", open("0005.npy", "rb")),
    "coronal":  ("coronal.npy",  open("0028.npy", "rb")),
    "axial":    ("axial.npy",    open("0014.npy", "rb")),
}
response = requests.post("https://sabnab-mrnet-api.hf.space/predict", files=files)
print(response.json())
```

### Example Response

```json
{
  "status": "success",
  "gatekeeper": "All planes validated",
  "files_processed": {
    "sagittal": "0005.npy",
    "coronal": "0028.npy",
    "axial": "0014.npy"
  },
  "predictions": {
    "ACL":      { "probability": 0.2698, "positive": false },
    "Meniscus": { "probability": 0.3585, "positive": false },
    "Abnormal": { "probability": 0.5689, "positive": true }
  },
  "clinical_summary": {
    "primary_finding": "Abnormal",
    "probability": 0.5689,
    "confidence_level": "Moderate Confidence",
    "positive_findings": ["Abnormal"],
    "recommendation": "Possible pathology detected. Consult a radiologist for clinical confirmation."
  },
  "inference_time_seconds": 11.114
}
```

### Gatekeeper Rejection Example

If the wrong plane is uploaded to the wrong field:

```json
{
  "detail": "Plane mismatch in '0014.npy': Expected 'sagittal' but received 'axial' (confidence: 94.2%). Please upload the correct MRI plane into the correct field."
}
```

---

##  Monitoring with Prometheus

The `/metrics` endpoint exposes real-time model telemetry:

| Metric | Description |
|--------|-------------|
| `model_predictions_total` | Total positive predictions per pathology |
| `model_inference_latency_seconds` | Inference time histogram |
| `gatekeeper_rejections_total` | Uploads rejected per expected plane |

---

## 🛠️ Tech Stack

| Layer              | Technology                              |
|--------------------|------------------------------------------|
| Model              | PyTorch · 3D CNN · 2D CNN (Gatekeeper)  |
| API                | FastAPI · Uvicorn                       |
| Monitoring         | Prometheus                              |
| Containerization   | Docker                                  |
| Deployment         | Hugging Face Spaces                     |
| CI/CD              | GitHub Actions                          |
| Large File Storage | Git LFS / Hugging Face Xet              |
| Language           | Python 3.10                             |

Author

Sabna Balasubramoniapillai M.S. Data Science, University of West Florida (GPA: 3.97) 
sabna.pillai@gmail.com ; GitHub Profile: (https://github.com/bSabna)

