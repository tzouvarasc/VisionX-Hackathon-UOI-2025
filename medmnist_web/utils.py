# utils.py
import json
from pathlib import Path
from typing import Dict, Any
import torch
from torchvision import transforms
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"

def load_metadata() -> Dict[str, Any]:
    with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

META = load_metadata()
IMG_SIZE = int(META.get("img_size", 28))
N_CHANNELS = int(META.get("n_channels", 3))
MEAN = META.get("mean", [0.5] * N_CHANNELS)
STD  = META.get("std",  [0.5] * N_CHANNELS)
CLASS_NAMES = META.get("class_names", [])
TASK = META.get("task", "multi-class")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_preprocess():
    # Αναγκαστική αλλαγή μεγέθους για να δέχεται και "ξένες" εικόνες
    tfms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    return transforms.Compose(tfms)

def pil_to_tensor(img: Image.Image, device: str):
    # Ανάλογα με τα κανάλια
    if N_CHANNELS == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    x = build_preprocess()(img).unsqueeze(0).to(device)  # (1, C, H, W)
    return x

@torch.no_grad()
def logits_to_output(logits: torch.Tensor) -> Dict[str, Any]:
    if TASK.startswith("multi-label"):
        probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()
        pred_idxs = [i for i,p in enumerate(probs) if p >= 0.5]
        pred_labels = [CLASS_NAMES[i] for i in pred_idxs] if CLASS_NAMES else pred_idxs
        return {"pred_class": pred_labels, "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}}
    else:
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top = int(probs.argmax())
        return {
            "pred_class": CLASS_NAMES[top] if CLASS_NAMES else top,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
        }
