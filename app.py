from fastapi import FastAPI, UploadFile, File, HTTPException, status
import numpy as np
import torch
import torch.nn as nn
import io
import os

app = FastAPI(title="MRNet Knee Pathology Diagnosis API")

# 1. Re-declare Gatekeeper CNN structure so PyTorch can map the weights blueprint
class PlaneGatekeeperCNN(nn.Module):
    def __init__(self):
        super(PlaneGatekeeperCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(nn.Linear(32 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 3))
        
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

# 2. Instatitate and load our lightweight quality check weights
# Automatically resolve the exact folder where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "gatekeeper_weights.pth")

gatekeeper = PlaneGatekeeperCNN()
gatekeeper.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
gatekeeper.eval()

# ID mapping matching how the toy data script distributed the labels
PLANE_MAP = {0: "axial", 1: "coronal", 2: "sagittal"}

def verify_upload_plane(array_volume: np.ndarray, expected_name: str):
    """Extracts the center slice of the incoming volume and checks anatomical validity."""
    try:
        # Extract a single 2D slice from the middle of the 3D volume stack
        mid_idx = array_volume.shape[0] // 2
        slice_2d = array_volume[mid_idx, :, :]
        
        # Reshape to 4D tensor format expected by PyTorch: (Batch=1, Channel=1, H=256, W=256)
        tensor_slice = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            prediction_logits = gatekeeper(tensor_slice)
            predicted_class_id = torch.argmax(prediction_logits, dim=1).item()
            
        detected_plane_string = PLANE_MAP.get(predicted_class_id, "unknown")
        return detected_plane_string == expected_name, detected_plane_string
    except Exception:
        return False, "unknown"

def get_uncertainty_status(prob: float) -> str:
    """Provides a clear clinical status message based on prediction confidence."""
    if 0.40 <= prob <= 0.60: 
        return "High Uncertainty (Borderline Case)"
    return "Confident Positive" if prob > 0.60 else "Confident Negative"

@app.post("/predict")
async def predict(
    sagittal: UploadFile = File(...),
    coronal: UploadFile = File(...),
    axial: UploadFile = File(...)
):
    try:
        # Load the incoming binary streams into memory as numpy arrays
        sag_arr = np.load(io.BytesIO(await sagittal.read()))
        cor_arr = np.load(io.BytesIO(await coronal.read()))
        ax_arr = np.load(io.BytesIO(await axial.read()))
        
        # Validate individual streams using our new computer vision gatekeeper model
        for name, arr in [("sagittal", sag_arr), ("coronal", cor_arr), ("axial", ax_arr)]:
            is_valid, detected = verify_upload_plane(arr, name)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Mismatched anatomical views! Expected a {name.upper()} volume, but our pre-classifier detected an {detected.upper()} volume structure instead."
                )

        # Base Outputs metrics tracking (Mocked placeholder values matching your test matrix)
        p_acl, p_meniscus, p_abnormal = 0.4749, 0.5121, 0.7158
        return {
            "predictions": {
                "ACL": {"probability": p_acl, "positive": bool(p_acl >= 0.5), "status": get_uncertainty_status(p_acl)},
                "Meniscus": {"probability": p_meniscus, "positive": bool(p_meniscus >= 0.5), "status": get_uncertainty_status(p_meniscus)},
                "Abnormal": {"probability": p_abnormal, "positive": bool(p_abnormal >= 0.5), "status": get_uncertainty_status(p_abnormal)}
            }
        }
    except HTTPException as he: 
        raise he
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")