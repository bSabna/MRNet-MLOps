import io
import time
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from prometheus_client import Counter, Histogram, make_asgi_app
 
from mrnet_architecture import load_production_model, preprocess_volume
 

# APP INIT

 
app = FastAPI(
    title="MRNet Knee Pathology Diagnosis API",
    description="End-to-end MLOps pipeline for knee MRI classification with plane validation.",
    version="2.0.0"
)
 
# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
 
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of positive predictions",
    ["pathology"]
)
LATENCY_HISTOGRAM = Histogram(
    "model_inference_latency_seconds",
    "Inference latency in seconds"
)
GATEKEEPER_REJECTION_COUNTER = Counter(
    "gatekeeper_rejections_total",
    "Number of uploads rejected by gatekeeper",
    ["expected_plane"]
)
 
 

# 1. GATEKEEPER MODEL

 
class PlaneGatekeeperCNN(torch.nn.Module):
    """Lightweight 2D CNN to verify MRI anatomical plane orientation."""
    def __init__(self):
        super(PlaneGatekeeperCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
 
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
 
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32 * 4 * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
 
 
# Label mapping: matches train_gatekeeper.py
PLANE_LABELS = {0: "axial", 1: "coronal", 2: "sagittal"}
 
print("Loading gatekeeper model...")
gatekeeper = PlaneGatekeeperCNN()
gatekeeper.load_state_dict(
    torch.load("gatekeeper_weights.pth", map_location=torch.device("cpu"))
)
gatekeeper.eval()
print(" Gatekeeper model loaded.")
 
 

# 2. MAIN DIAGNOSIS MODEL

print("Loading main 3D CNN diagnosis model...")
MODEL_PATH = "mrnet_3dcnn_artifacts.pth"
diagnosis_model = load_production_model(MODEL_PATH)
print(" Diagnosis model loaded.")
 
# Optimal thresholds from Youden's J statistic
THRESHOLDS = {"ACL": 0.356, "Meniscus": 0.400, "Abnormal": 0.497}
 
 

# 3. HELPER FUNCTIONS

 
def bytes_to_middle_slice_tensor(file_bytes: bytes) -> torch.Tensor:
    """
    Load .npy volume from bytes, extract middle slice,
    normalize, resize to 256x256, return (1, 1, 256, 256) tensor.
    """
    arr = np.load(io.BytesIO(file_bytes))
 
    # Extract middle slice from 3D volume
    if len(arr.shape) == 3:
        mid = arr.shape[0] // 2
        slice_2d = arr[mid].astype(np.float32)
    elif len(arr.shape) == 2:
        slice_2d = arr.astype(np.float32)
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")
 
    # Normalize to [0, 1]
    vmin, vmax = slice_2d.min(), slice_2d.max()
    if vmax > vmin:
        slice_2d = (slice_2d - vmin) / (vmax - vmin)
 
    # Resize to 256x256
    from PIL import Image
    img = Image.fromarray(slice_2d).resize((256, 256), Image.BILINEAR)
    slice_2d = np.array(img)
 
    return torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
 
 
def validate_plane(file_bytes: bytes, expected_plane: str) -> tuple[bool, str, float]:
    """
    Run gatekeeper on a single file.
    Returns (is_valid, predicted_plane, confidence)
    """
    tensor = bytes_to_middle_slice_tensor(file_bytes)
    with torch.no_grad():
        logits = gatekeeper(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        predicted_idx = probs.argmax().item()
        predicted_plane = PLANE_LABELS[predicted_idx]
        confidence = float(probs[predicted_idx])
 
    is_valid = (predicted_plane == expected_plane)
    return is_valid, predicted_plane, confidence
 
 

# 4. ENDPOINTS

 
@app.get("/", summary="Root")
def root():
    return {"status": "healthy", "message": "MRNet Knee Pathology API is operational."}
 
 
@app.get("/health", summary="Health Check")
def health():
    return {"status": "healthy"}
 
 
@app.post("/predict", summary="Diagnose Knee Pathology from MRI Views")
async def predict(
    sagittal: UploadFile = File(..., description="Sagittal plane MRI (.npy)"),
    coronal:  UploadFile = File(..., description="Coronal plane MRI (.npy)"),
    axial:    UploadFile = File(..., description="Axial plane MRI (.npy)")
):
    """
    Step 1 — Gatekeeper validates each upload is the correct anatomical plane.
    Step 2 — Main 3D CNN runs multi-label diagnosis.
    Returns predictions for ACL tear, Meniscus tear, and Abnormality.
    """
    start_time = time.time()
 
    # Read file bytes
    sag_bytes = await sagittal.read()
    cor_bytes = await coronal.read()
    axi_bytes = await axial.read()
 
    # ── GATEKEEPER VALIDATION ──
    validations = [
        (sag_bytes, "sagittal", sagittal.filename),
        (cor_bytes, "coronal",  coronal.filename),
        (axi_bytes, "axial",    axial.filename),
    ]
 
    for file_bytes, expected_plane, filename in validations:
        is_valid, predicted_plane, confidence = validate_plane(file_bytes, expected_plane)
        if not is_valid:
            GATEKEEPER_REJECTION_COUNTER.labels(expected_plane=expected_plane).inc()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Plane mismatch detected in '{filename}': "
                    f"Expected '{expected_plane}' scan but received '{predicted_plane}' "
                    f"(confidence: {confidence:.1%}). "
                    f"Please upload the correct MRI plane into the correct field."
                )
            )
 
    # ── MAIN DIAGNOSIS ──
    try:
        sag_tensor = preprocess_volume(np.load(io.BytesIO(sag_bytes)))
        cor_tensor = preprocess_volume(np.load(io.BytesIO(cor_bytes)))
        axi_tensor = preprocess_volume(np.load(io.BytesIO(axi_bytes)))
 
        with torch.no_grad():
            logits = diagnosis_model(sag_tensor, cor_tensor, axi_tensor)
            probs = torch.sigmoid(logits).squeeze(0).numpy()
 
        results = {
            "ACL":      {"probability": round(float(probs[0]), 4), "positive": bool(probs[0] > THRESHOLDS["ACL"])},
            "Meniscus": {"probability": round(float(probs[1]), 4), "positive": bool(probs[1] > THRESHOLDS["Meniscus"])},
            "Abnormal": {"probability": round(float(probs[2]), 4), "positive": bool(probs[2] > THRESHOLDS["Abnormal"])},
        }
 
        # Prometheus counters
        for pathology, data in results.items():
            if data["positive"]:
                PREDICTION_COUNTER.labels(pathology=pathology).inc()
 
        LATENCY_HISTOGRAM.observe(time.time() - start_time)
 
        return {
            "status": "success",
            "gatekeeper": "All planes validated ",
            "files_processed": {
                "sagittal": sagittal.filename,
                "coronal":  coronal.filename,
                "axial":    axial.filename,
            },
            "predictions": results,
            "inference_time_seconds": round(time.time() - start_time, 3)
        }
 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )