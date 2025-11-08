# app_fastapi.py
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from pathlib import Path
from utils import get_device, pil_to_tensor, logits_to_output

app = FastAPI(title="PathMNIST CNN API", version="1.0.0")

# CORS: βάλε εδώ το frontend origin σου (στην παραγωγή ΜΗΝ αφήνεις "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # π.χ. ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = get_device()
MODEL_PATH = Path(__file__).parent / "models" / "pathmnist_cnn.ts"
MODEL = torch.jit.load(str(MODEL_PATH), map_location=DEVICE).eval()

@app.get("/")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Παρακαλώ ανέβασε αρχείο εικόνας.")
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Μη έγκυρη εικόνα.")
    x = pil_to_tensor(img, DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
    out = logits_to_output(logits)
    # Για απλό UI που θέλει ΚΕΙΜΕΝΟ, μπορείς να χρησιμοποιήσεις μόνο out["pred_class"]
    return out
