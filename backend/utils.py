import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def encode_rgb_to_base64(img_rgb):
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_image(image: Image.Image):
    """
    Preprocess PIL Image for TorchXRayVision model
    
    Args:
        image: PIL Image in RGB format
    
    Returns:
        img_tensor: Preprocessed tensor [1, 1, 224, 224]
        img_rgb: RGB image for visualization [224, 224, 3]
    """
    # Convert to grayscale
    img_gray = np.array(image.convert('L'))
    
    # Resize to 224x224
    img_resized = Image.fromarray(img_gray).resize((224, 224), Image.BILINEAR)
    img_array = np.array(img_resized).astype(np.float32)
    
    # Normalize for TorchXRayVision [-1024, 1024]
    img_normalized = img_array / 255.0
    img_normalized = (img_normalized - 0.5) * 2 * 1024
    
    # Convert to tensor [1, 1, 224, 224]
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    
    # Create RGB version for visualization [224, 224, 3]
    img_rgb = np.stack([img_array / 255.0] * 3, axis=-1)
    
    return img_tensor, img_rgb


def create_heatmap_images(img_rgb: np.ndarray, grayscale_cam: np.ndarray):
    """
    Create overlay and heatmap images from Grad-CAM output
    
    Args:
        img_rgb: RGB image [H, W, 3] normalized to [0, 1]
        grayscale_cam: Grad-CAM heatmap [H, W] normalized to [0, 1]
    
    Returns:
        overlay_b64: Base64 encoded overlay image
        heatmap_b64: Base64 encoded heatmap image
    """
    # Create overlay
    overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True, image_weight=0.6)
    
    # Create heatmap visualization
    heatmap = create_colored_heatmap(grayscale_cam)
    
    # Convert to base64
    overlay_b64 = pil_to_base64(Image.fromarray(overlay))
    heatmap_b64 = pil_to_base64(Image.fromarray(heatmap))
    
    return overlay_b64, heatmap_b64


def create_colored_heatmap(grayscale_cam: np.ndarray):
    """
    Create a colored heatmap from grayscale Grad-CAM
    
    Args:
        grayscale_cam: Grad-CAM heatmap [H, W] normalized to [0, 1]
    
    Returns:
        heatmap_rgb: RGB heatmap image [H, W, 3]
    """
    # Apply colormap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(grayscale_cam)[:, :, :3]  # Remove alpha channel
    heatmap_rgb = (heatmap * 255).astype(np.uint8)
    
    return heatmap_rgb


def pil_to_base64(image: Image.Image):
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image
    
    Returns:
        base64_str: Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def create_prediction_bars(probs: dict, top_n: int = 5):
    """
    Create a bar chart visualization of top predictions
    
    Args:
        probs: Dictionary of {pathology: probability}
        top_n: Number of top predictions to show
    
    Returns:
        chart_b64: Base64 encoded bar chart image
    """
    # Sort and get top N
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color='#7c3aed')
    
    # Styling
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffered = io.BytesIO()
    plt.savefig(buffered, format="PNG", dpi=100, bbox_inches='tight')
    plt.close(fig)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str