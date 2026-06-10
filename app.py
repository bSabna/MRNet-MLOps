import io
import torch
import torch.nn as nn
import numpy as np
import base64
import os
import scipy.stats  # Used to compute entropy for the uncertainty summary
from fastapi import FastAPI, UploadFile, File, HTTPException, status

app = FastAPI(title="MRNet Knee Pathology Diagnosis API")

# 1. Fetch the giant string securely from Hugging Face Secrets
WEIGHTS_BASE64 = os.environ.get("GATEKEEPER_WEIGHTS")

if not WEIGHTS_BASE64:
    raise RuntimeError(
        "CRITICAL ERROR: 'GATEKEEPER_WEIGHTS' secret not found in Hugging Face Space Settings!"
    )

# 2. Reconstructed Model Class matching your exact state_dict shapes
class GatekeeperModel(nn.Module):
    def __init__(self):
        super(GatekeeperModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32), 
            nn.ReLU(),
            nn.Linear(32, 3) 
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 3. Decode and reconstruct the weights in-memory during container startup
try:
    print("Decoding model weights from Hugging Face Secrets...")
    weights_binary = base64.b64decode(WEIGHTS_BASE64)
    buffer = io.BytesIO(weights_binary)
    
    model = GatekeeperModel() 
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights initialized successfully out of secure memory!")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise e


def preprocess_tensor(file_bytes: bytes) -> torch.Tensor:
    """
    Parses raw bytes into a NumPy array, extracts a representative slice, 
    and handles single-channel matching for the model.
    """
    # Load numpy array directly out of the binary byte memory stream
    arr = np.load(io.BytesIO(file_bytes))
    
    # If the array is a 3D volume (Slices, Height, Width), let's extract the middle slice
    if len(arr.shape) == 3:
        middle_idx = arr.shape[0] // 2
        tensor_2d = arr[middle_idx]
    elif len(arr.shape) == 2:
        tensor_2d = arr
    else:
        # Fallback/reshape if shape is unconventional
        tensor_2d = arr.reshape(arr.shape[-2], arr.shape[-1])
        
    # Convert to float PyTorch tensor, add Batch and Channel dimensions: (1, 1, H, W)
    tensor = torch.tensor(tensor_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


@app.get("/")
def home():
    return {"status": "healthy", "message": "MRNet API is operational."}


@app.post("/predict", summary="Diagnose Knee Pathology from MRI Views")
async def predict(
    axial_file: UploadFile = File(..., description="The Axial view MRI tensor data (.npy file)"),
    sagittal_file: UploadFile = File(..., description="The Sagittal view MRI tensor data (.npy file)"),
    coronal_file: UploadFile = File(..., description="The Coronal view MRI tensor data (.npy file)")
):
    """
    Processes Axial, Sagittal, and Coronal views dynamically, evaluates them 
    using the model, and builds a comprehensive pathology and uncertainty summary.
    """
    try:
        # Read uploaded file streams
        axial_bytes = await axial_file.read()
        sagittal_bytes = await sagittal_file.read()
        coronal_bytes = await coronal_file.read()
        
        # Preprocess each view to structural (1, 1, H, W) tensors
        t_axial = preprocess_tensor(axial_bytes)
        t_sagittal = preprocess_tensor(sagittal_bytes)
        t_coronal = preprocess_tensor(coronal_bytes)
        
        # Combine view representations via mean aggregation for the prediction profile
        with torch.no_grad():
            out_axial = model(t_axial)
            out_sagittal = model(t_sagittal)
            out_coronal = model(t_coronal)
            
            # Combine raw logits across all three perspectives
            combined_logits = (out_axial + out_sagittal + out_coronal) / 3.0
            probabilities = torch.softmax(combined_logits, dim=1).squeeze().tolist()
            
        # Map probabilities out to the 3 distinct classes
        class_mappings = ["Abnormal", "ACL Tear", "Meniscus Tear"]
        predictions_dict = {class_mappings[i]: round(probabilities[i], 4) for i in range(3)}
        
        # Compute Shannon Entropy to determine diagnostic uncertainty
        # A uniform distribution across 3 classes yields maximum entropy (~1.098)
        entropy_val = scipy.stats.entropy(probabilities)
        max_entropy = np.log(3)
        uncertainty_score = float(entropy_val / max_entropy)
        
        # Determine human-readable uncertainty description strings
        if uncertainty_score > 0.65:
            uncertainty_summary = "Highly Uncertain. Model predictions are split closely across multiple conditions. Clinical verification strongly recommended."
        elif uncertainty_score > 0.35:
            uncertainty_summary = "Moderately Uncertain. Clear lean toward primary diagnosis, but secondary indicators are active."
        else:
            uncertainty_summary = "Highly Confident. Strong mathematical convergence toward the leading diagnosis profile."

        return {
            "status": "success",
            "files_processed": {
                "axial": axial_file.filename,
                "sagittal": sagittal_file.filename,
                "coronal": coronal_file.filename
            },
            "predictions": predictions_dict,
            "uncertainty_assessment": {
                "normalized_uncertainty_score": round(uncertainty_score, 4),
                "summary": uncertainty_summary
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating patient scans: {str(e)}"
        )