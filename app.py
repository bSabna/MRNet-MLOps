import time
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

# FIXED: Pointing directly to your flat architecture file
from mrnet_architecture import load_production_model, preprocess_volume

app = FastAPI(title="MRNet Knee Pathology Diagnosis API")

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Prometheus Tracking Metrics
PREDICTION_COUNTER = Counter("model_predictions_total", "Total number of predictions", ["pathology"])
LATENCY_HISTOGRAM = Histogram("model_inference_latency_seconds", "Inference latency in seconds")

# FIXED: Removed the trailing space inside the string
MODEL_PATH = "mrnet_3dcnn_artifacts.pth"
model = load_production_model(MODEL_PATH)

# Thresholds calculated using Youden's J Statistic during training
THRESHOLDS = {"ACL": 0.356, "Meniscus": 0.400, "Abnormal": 0.497}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(
    sagittal: UploadFile = File(...), 
    coronal: UploadFile = File(...), 
    axial: UploadFile = File(...)
):
    start_time = time.time()
    try:
        # 1. Load uploaded binary .npy streams
        sag_data = np.load(sagittal.file)
        cor_data = np.load(coronal.file)
        axi_data = np.load(axial.file)
        
        # 2. Preprocess tensors
        sag_tensor = preprocess_volume(sag_data)
        cor_tensor = preprocess_volume(cor_data)
        axi_tensor = preprocess_volume(axi_data)
        
        # 3. Model Inference
        with torch.no_grad():
            logits = model(sag_tensor, cor_tensor, axi_tensor)
            probs = torch.sigmoid(logits).squeeze(0).numpy() # [3]
            
        # 4. Map probabilities to classes using optimal thresholds
        results = {
            "ACL": {"probability": float(probs[0]), "positive": bool(probs[0] > THRESHOLDS["ACL"])},
            "Meniscus": {"probability": float(probs[1]), "positive": bool(probs[1] > THRESHOLDS["Meniscus"])},
            "Abnormal": {"probability": float(probs[2]), "positive": bool(probs[2] > THRESHOLDS["Abnormal"])}
        }
        
        # Log Prometheus Counter metrics
        for pathology, data in results.items():
            if data["positive"]:
                PREDICTION_COUNTER.labels(pathology=pathology).inc()
                
        LATENCY_HISTOGRAM.observe(time.time() - start_time)
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")