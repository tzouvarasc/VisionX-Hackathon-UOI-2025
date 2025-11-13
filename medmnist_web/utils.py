"""Utility helpers for the Chest X-ray demo."""
import base64
import io
from typing import Dict, Any, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

IMG_SIZE = 224
DEFAULT_CLASS_NAMES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "Fracture",
    "Lung_Lesion",
    "Lung_Opacity",
    "Pleural_Other",
]
CLASS_NAMES: List[str] = DEFAULT_CLASS_NAMES.copy()
DISPLAY_NAMES: List[str] = [name.replace("_", " ") for name in CLASS_NAMES]

_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC)
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(mean=[0.485], std=[0.229])


def set_class_names(names: Iterable[str]) -> None:
    """Update the class name mapping using the model metadata."""
    global CLASS_NAMES, DISPLAY_NAMES
    normalized = list(names)
    if not normalized:
        normalized = DEFAULT_CLASS_NAMES.copy()
    CLASS_NAMES = normalized
    DISPLAY_NAMES = [name.replace("_", " ") for name in CLASS_NAMES]


def get_display_names() -> List[str]:
    return DISPLAY_NAMES


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_densenet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last convolutional block for DenseNet-based models."""
    candidate = getattr(model, "model", model)
    features = getattr(candidate, "features", None)
    if features is None:
        raise RuntimeError("Unable to resolve DenseNet features for Grad-CAM.")
    if hasattr(features, "denseblock4"):
        return features.denseblock4
    children = list(features.children())
    if not children:
        raise RuntimeError("Model features have no child layers to hook.")
    return children[-1]


def pil_to_tensor(img: Image.Image, device: str) -> Tuple[torch.Tensor, Image.Image]:
    """Prepare an image for the model and return the tensor plus the resized PIL copy."""
    img = img.convert("L")  # chest X-rays are grayscale
    resized = _RESIZE(img)
    tensor = _TO_TENSOR(resized)
    tensor_for_model = tensor.clone()
    tensor_for_model = _NORMALIZE(tensor_for_model)
    tensor_for_model = tensor_for_model.unsqueeze(0).to(device)
    return tensor_for_model, resized.convert("RGB")


def logits_to_output(logits: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
    probs = torch.sigmoid(logits)[0]
    probs_np = probs.detach().cpu().numpy()
    sorted_idxs = np.argsort(-probs_np)
    activated = [i for i, p in enumerate(probs_np) if p >= threshold]
    if not activated:
        activated = [int(sorted_idxs[0])]
    labels = [DISPLAY_NAMES[i] for i in activated]
    return {
        "pred_class": labels,
        "probs": {DISPLAY_NAMES[i]: float(probs_np[i]) for i in range(len(probs_np))},
        "top_indices": [int(i) for i in sorted_idxs[:3]],
    }


class GradCAM:
    """Lightweight Grad-CAM helper for PyTorch models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, _inp, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, target_idx: int | None = None) -> Tuple[torch.Tensor, np.ndarray, int]:
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        if target_idx is None:
            target_idx = int(probs[0].argmax().item())
        score = probs[:, target_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        cam = self._build_cam()
        self.model.zero_grad(set_to_none=True)
        return logits, cam, target_idx

    def _build_cam(self) -> np.ndarray:
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks were not called. Ensure a forward/backward pass happened.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = cam.detach().cpu()
        cam -= cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam /= max_val
        return cam.numpy()

    def close(self) -> None:
        if getattr(self, "_forward_handle", None) is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if getattr(self, "_backward_handle", None) is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def __del__(self):
        self.close()


_COLOR_STOPS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 128.0],
    [0.0, 128.0, 255.0],
    [255.0, 255.0, 0.0],
    [255.0, 128.0, 0.0],
    [255.0, 0.0, 0.0],
])
_POSITIONS = np.linspace(0, 1, len(_COLOR_STOPS))


def _colormap(cam: np.ndarray) -> Image.Image:
    cam = np.clip(cam, 0.0, 1.0)
    r = np.interp(cam, _POSITIONS, _COLOR_STOPS[:, 0])
    g = np.interp(cam, _POSITIONS, _COLOR_STOPS[:, 1])
    b = np.interp(cam, _POSITIONS, _COLOR_STOPS[:, 2])
    heatmap = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(heatmap, mode="RGB")


def render_gradcam_overlay(base_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Tuple[str, str]:
    """Create overlay and heatmap images encoded as base64 strings."""
    heatmap = _colormap(cam)
    heatmap = heatmap.resize(base_image.size, Image.BILINEAR)
    overlay = Image.blend(base_image.convert("RGB"), heatmap, alpha)
    return image_to_base64(overlay), image_to_base64(heatmap)


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
