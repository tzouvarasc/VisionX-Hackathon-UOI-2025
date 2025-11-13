# app_fastapi.py
import io
from typing import Any

import torchxrayvision as xrv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

if __package__:
    from .utils import (
        GradCAM,
        get_device,
        get_display_names,
        logits_to_output,
        pil_to_tensor,
        render_gradcam_overlay,
        resolve_densenet_target_layer,
        set_class_names,
    )
else:  # pragma: no cover - executed when running as a top-level module
    from utils import (
        GradCAM,
        get_device,
        get_display_names,
        logits_to_output,
        pil_to_tensor,
        render_gradcam_overlay,
        resolve_densenet_target_layer,
        set_class_names,
    )

app = FastAPI(title="CheX DenseNet API", version="2.0.0")

# CORS: βάλε εδώ το frontend origin σου (στην παραγωγή ΜΗΝ αφήνεις "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # π.χ. ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = get_device()
MODEL = xrv.models.DenseNet(weights="densenet121-res224-chex").to(DEVICE)
MODEL.eval()
set_class_names(getattr(MODEL, "pathologies", getattr(MODEL, "classes", [])))
GRADCAM = GradCAM(MODEL, resolve_densenet_target_layer(MODEL))

@app.get("/")
def health():
    return {"status": "ok", "device": DEVICE, "classes": get_display_names()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Παρακαλώ ανέβασε αρχείο εικόνας.")
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Μη έγκυρη εικόνα.")
    x, resized = pil_to_tensor(img, DEVICE)
    logits, cam, target_idx = GRADCAM(x)
    response: dict[str, Any] = logits_to_output(logits)
    overlay_b64, heatmap_b64 = render_gradcam_overlay(resized, cam)
    response.pop("top_indices", None)
    response["gradcam_overlay"] = overlay_b64
    response["gradcam_heatmap"] = heatmap_b64
    display_names = get_display_names()
    if 0 <= target_idx < len(display_names):
        response["gradcam_target"] = display_names[target_idx]
    # Για απλό UI που θέλει ΚΕΙΜΕΝΟ, μπορείς να χρησιμοποιήσεις μόνο response["pred_class"]
    return response
