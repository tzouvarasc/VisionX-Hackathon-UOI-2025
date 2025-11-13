# Hackathon – Chest X-ray Demo

Interactive web experience for exploring the pretrained TorchXRayVision DenseNet121 (CheX) model with Grad-CAM explanations.

## Setup

```bash
pip install -r medmnist_web/requirements.txt
```

## Run the FastAPI backend

```bash
uvicorn medmnist_web.app_fastapi:app --host 0.0.0.0 --port 8080 --reload
```

## Serve the UI

```bash
cd medmnist_web/static
python -m http.server 5500
```

Then visit `http://localhost:5500` and upload a chest X-ray to inspect predictions and Grad-CAM overlays.
