# Hackathon – Chest X-ray Demo

Interactive web experience for exploring the pretrained TorchXRayVision DenseNet121 (CheX) model with Grad-CAM explanations.

## Setup

```bash
pip install -r medmnist_web/requirements.txt
```

## Run the FastAPI backend

```bash
cd medmnist_web
uvicorn app_fastapi:app --host 0.0.0.0 --port 8080 --reload
```

> Αν προτιμάς να μείνεις στον ριζικό φάκελο του repo, τρέξε `uvicorn medmnist_web.app_fastapi:app --host 0.0.0.0 --port 8080 --reload`. Σε κάθε περίπτωση βεβαιώσου ότι το working directory περιέχει τον φάκελο `medmnist_web`, αλλιώς η Python δεν μπορεί να εντοπίσει το πακέτο.

## Serve the UI

```bash
cd medmnist_web/static
python -m http.server 5500
```

Then visit `http://localhost:5500` and upload a chest X-ray to inspect predictions and Grad-CAM overlays.
