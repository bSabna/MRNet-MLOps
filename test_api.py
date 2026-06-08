import numpy as np
import requests
import json
import os

print(" Step 1: Generating dummy 3D MRI scans...")
dummy_sag = np.random.randint(0, 255, (32, 256, 256)).astype(np.float32)
dummy_cor = np.random.randint(0, 255, (32, 256, 256)).astype(np.float32)
dummy_axi = np.random.randint(0, 255, (32, 256, 256)).astype(np.float32)

np.save("sag.npy", dummy_sag)
np.save("cor.npy", dummy_cor)
np.save("axi.npy", dummy_axi)

print("Step 2: Packaging files into a multi-part form upload...")
# Using 'with' safely opens and closes the files automatically
with open("sag.npy", "rb") as s, open("cor.npy", "rb") as c, open("axi.npy", "rb") as a:
    payload_files = {
        "sagittal": ("sag.npy", s, "application/octet-stream"),
        "coronal": ("cor.npy", c, "application/octet-stream"),
        "axial": ("axi.npy", a, "application/octet-stream")
    }

    print("Step 3: Transmitting arrays to container at http://localhost:8000/predict...")
    target_url = "http://localhost:8000/predict"
    
    try:
        response = requests.post(target_url, files=payload_files)
        print("\n --- API RESPONSE RECEIVED ---")
        print(f"HTTP Status Code: {response.status_code}")
        print("Model Diagnosis Output JSON:")
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"\n Prediction failed! Error: {e}")

# Windows can now delete them safely because the 'with' block closed them!
print(" Cleaning up temporary test files...")
for f in ["sag.npy", "cor.npy", "axi.npy"]:
    if os.path.exists(f): 
        os.remove(f)