import io
import torch
import torch.nn as nn
import numpy as np
import base64
import os
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
            # CHANGED: Changed 2 to 3 to match your trained checkpoint matrix
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
    
    # Initialize our reconstructed model framework
    model = GatekeeperModel() 
    
    # Load the state dictionary directly into CPU memory
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights initialized successfully out of secure memory!")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise e


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
    Submit the 3 key MRI sequences (Axial, Sagittal, Coronal) 
    to get the multi-class pathology predictions.
    """
    try:
        # 1. Read the raw bytes from the uploaded files
        axial_bytes = await axial_file.read()
        sagittal_bytes = await sagittal_file.read()
        coronal_bytes = await coronal_file.read()
        
        # TODO: Implement your preprocessing pipeline here 
        # e.g., np.load(io.BytesIO(axial_bytes)), tracking tensors, etc.
        
        # Placeholder dictionary representing your 3 target classes
        return {
            "status": "success",
            "filename_received": {
                "axial": axial_file.filename,
                "sagittal": sagittal_file.filename,
                "coronal": coronal_file.filename
            },
            "predictions": {
                "Abnormal": 0.85,
                "ACL Tear": 0.12,
                "Meniscus Tear": 0.03
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}"
        )