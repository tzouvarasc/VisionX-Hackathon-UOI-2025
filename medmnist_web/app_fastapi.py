# app_fastapi.py
import io
from pathlib import Path

import torch
import torchxrayvision as xrv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from utils import (
    CLASS_NAMES,
    gradcam_to_base64,
    get_device,
    logits_to_output,
    pil_to_tensor,
)

app = FastAPI(title="CheXpert DenseNet API", version="1.0.0")

# CORS: βάλε εδώ το frontend origin σου (στην παραγωγή ΜΗΝ αφήνεις "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # π.χ. ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = get_device()
MODELS_DIR = Path(__file__).parent / "models"
WEIGHTS_NAME = "densenet121-res224-chex"


def load_model(device: str) -> torch.nn.Module:
    model = xrv.models.DenseNet(weights=WEIGHTS_NAME)
    model.to(device)
    model.eval()
    return model


MODEL = load_model(DEVICE)
TARGET_LAYER = MODEL.features.denseblock4

@app.get("/")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Παρακαλώ ανέβασε αρχείο εικόνας.")
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Μη έγκυρη εικόνα.")
    x = pil_to_tensor(img, DEVICE)
    logits, heatmap_b64 = gradcam_to_base64(MODEL, x, TARGET_LAYER, img)
    out = logits_to_output(logits)
    if heatmap_b64 is not None:
        out["heatmap"] = heatmap_b64
    # Για απλό UI που θέλει ΚΕΙΜΕΝΟ, μπορείς να χρησιμοποιήσεις μόνο out["pred_class"]
    return out
