# utils.py
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image, ImageOps
from torchvision import transforms

MODELS_DIR = Path(__file__).parent / "models"

def load_metadata() -> Dict[str, Any]:
    with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)

META = load_metadata()
IMG_SIZE = int(META.get("img_size", 28))
N_CHANNELS = int(META.get("n_channels", 3))
MEAN = META.get("mean", [0.5] * N_CHANNELS)
STD = META.get("std", [0.5] * N_CHANNELS)
CLASS_NAMES = META.get("class_names", [])
TASK = META.get("task", "multi-class")
PRETRAINED_SOURCE = META.get("pretrained_source")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_preprocess():
    if PRETRAINED_SOURCE == "torchxrayvision":
        # TorchXRayVision περιμένει μονόκανάλικες εικόνες κλιμακωμένες στο [0, 1].
        def _xrv_preprocess(image: Image.Image) -> torch.Tensor:
            gray = image.convert("L")
            arr = np.array(gray)
            arr = xrv.datasets.XRayCenterCrop()(arr)
            arr = xrv.datasets.XRayResizer(IMG_SIZE)(arr)
            arr = arr.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)
            return tensor

        return _xrv_preprocess

    # Αναγκαστική αλλαγή μεγέθους για να δέχεται και "ξένες" εικόνες
    tfms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    return transforms.Compose(tfms)

def pil_to_tensor(img: Image.Image, device: str):
    preprocess = build_preprocess()
    if PRETRAINED_SOURCE == "torchxrayvision":
        tensor = preprocess(img)
        x = tensor.unsqueeze(0).to(device)
    else:
        if N_CHANNELS == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)  # (1, C, H, W)
    return x


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _register_backward_hook(module, hook):
    if hasattr(module, "register_full_backward_hook"):
        return module.register_full_backward_hook(hook)
    return module.register_backward_hook(lambda m, grad_in, grad_out: hook(m, grad_in, grad_out))


def compute_grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_index: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, _inp, output):
        activations.append(output.detach())

    def backward_hook(module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = _register_backward_hook(target_layer, backward_hook)

    input_tensor = input_tensor.requires_grad_(True)
    logits = model(input_tensor)

    if target_index is None:
        if TASK.startswith("multi-label"):
            probs = torch.sigmoid(logits)[0]
            target_index = int(torch.argmax(probs).item())
        else:
            target_index = int(torch.argmax(logits, dim=1).item())

    model.zero_grad(set_to_none=True)
    score = logits[:, target_index]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    if not activations or not gradients:
        return logits.detach(), None

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    cam = cam.squeeze().cpu()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return logits.detach(), cam.numpy()


def build_heatmap_overlay(base_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    resized = base_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    heat = Image.fromarray(np.uint8(heatmap * 255), mode="L")
    heat = heat.resize(resized.size, resample=Image.BILINEAR)
    colorized = ImageOps.colorize(heat, black="#0b1f3a", white="#f97316")
    overlay = Image.blend(resized, colorized, alpha)
    return overlay


def gradcam_to_base64(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    original_image: Image.Image,
    target_index: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[str]]:
    logits, heatmap = compute_grad_cam(model, input_tensor, target_layer, target_index)
    if heatmap is None:
        return logits, None
    overlay = build_heatmap_overlay(original_image, heatmap)
    return logits, image_to_base64(overlay)

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
