# no need for now

from fastapi import FastAPI, UploadFile, File
import torch  # Active
import numpy as np
import io
from mrnet_architecture import Improved3DCNN  # Active

app = FastAPI(title="MRNet 3D CNN API")

# --- ACTIVE FOR DOCKER PRODUCTION ---
model = Improved3DCNN()
checkpoint = torch.load('mrnet_3dcnn_artifacts.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

@app.get("/")
def home():
    return {"message": "MRNet API is Running"}

@app.post("/predict")
async def predict(
    sagittal: UploadFile = File(...), 
    coronal: UploadFile = File(...), 
    axial: UploadFile = File(...)
):
    # 1. Read the bytes from the uploaded files
    sag_data = np.load(io.BytesIO(await sagittal.read())).astype(np.float32)
    cor_data = np.load(io.BytesIO(await coronal.read())).astype(np.float32)
    axi_data = np.load(io.BytesIO(await axial.read())).astype(np.float32)

    # 2. Preprocess (Z-score and padding)
    def prep(vol):
        vol = (vol - vol.mean()) / (vol.std() + 1e-5)
        # Standardize to 32 slices as per your training
        if vol.shape[0] >= 32: vol = vol[:32]
        else: vol = np.pad(vol, ((0, 32-vol.shape[0]), (0,0), (0,0)))
        return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)

    # 3. Inference
    with torch.no_grad():
        output = model(prep(sag_data), prep(cor_data), prep(axi_data))
        probs = torch.sigmoid(output).numpy()[0]

    # 4. Return JSON results
    return {
        "ACL": float(probs[0]),
        "Meniscus": float(probs[1]),
        "Abnormal": float(probs[2])
    }