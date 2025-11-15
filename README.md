# CheXpert DenseNet Web App 🏥

AI-powered chest X-ray analysis with DenseNet121 and Grad-CAM visualizations.

## 📁 Project Structure

```
chexpert-app/
├── backend/
│   ├── app_fastapi.py      # FastAPI server
│   ├── utils.py            # Helper functions
│   └── requirements.txt    # Dependencies
├── frontend/
│   ├── index.html          # Web UI
│   └── style.css           # Styles
└── README.md
```

## 🚀 Setup & Installation

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Backend

```bash
# From backend directory
python app_fastapi.py

# Or with uvicorn directly:
uvicorn app_fastapi:app --host 0.0.0.0 --port 8080 --reload
```

Server θα τρέξει στο: **http://localhost:8080**

### 3. Run Frontend

Δύο επιλογές:

#### Option A: Python HTTP Server (Quick)
```bash
# Navigate to frontend directory
cd frontend

# Python 3
python -m http.server 3000

# Frontend θα είναι διαθέσιμο στο: http://localhost:3000
```

#### Option B: Live Server (VS Code)
1. Εγκατέστησε το "Live Server" extension στο VS Code
2. Right-click στο `index.html` → "Open with Live Server"

### 4. Access the App

Άνοιξε το browser και πήγαινε στο frontend URL (π.χ. http://localhost:3000)

## 🎯 Usage

### Single Image Analysis

1. **Upload X-ray**: Select "Single Image Analysis" mode, then drag & drop or click "Select Image"
2. **Predict**: Click "Predict" button
3. **View Results**: 
   - Top 5 pathology predictions
   - Probability bars
   - Grad-CAM heatmaps (overlay & pure heatmap)
   - AI-generated explanation

### Batch Risk Classification (NEW!)

1. **Select Mode**: Choose "Batch Risk Classification" mode
2. **Upload Multiple Images**: Select multiple chest X-ray images at once
3. **Classify**: Click "Predict" to process all images
4. **View Results**: Images are automatically classified into risk categories:
   - 🔴 **High Risk**: Consolidation, Pneumothorax, Edema, Pneumonia, Lung Lesion
   - 🟡 **Medium Risk**: Atelectasis, Effusion, Cardiomegaly, Enlarged Cardiomediastinum, Lung Opacity
   - 🟢 **Low Risk**: Fracture
5. **Output**: Images are saved to `classified_images/` folder organized by risk level

## 🔌 API Endpoints

### `GET /`
Health check endpoint
```json
{
  "status": "online",
  "model": "DenseNet121-CheXpert",
  "device": "cuda",
  "pathologies": [...]
}
```

### `POST /predict`
Upload chest X-ray for single image prediction

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "pred_class": ["Cardiomegaly", "Edema", ...],
  "probs": {
    "Atelectasis": 0.234,
    "Cardiomegaly": 0.876,
    ...
  },
  "gradcam_overlay": "base64_string...",
  "gradcam_heatmap": "base64_string...",
  "gradcam_target": "Cardiomegaly"
}
```

### `POST /predict_batch` (NEW!)
Upload multiple chest X-rays for batch risk classification

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `files` (multiple image files)

**Response:**
```json
{
  "results": [
    {
      "filename": "xray1.jpg",
      "top_label": "Pneumonia",
      "top_probability": 0.87,
      "risk_level": "High",
      "top_predictions": [...],
      "saved_path": "classified_images/High/0000_xray1.jpg"
    },
    ...
  ],
  "summary": {
    "High": 3,
    "Medium": 5,
    "Low": 2
  },
  "output_directory": "classified_images"
}
```

### `POST /llm_explain`
Get AI-generated explanation based on predictions

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body: 
```json
{
  "pred_class": ["Cardiomegaly", "Edema"],
  "probs": {...}
}
```

**Response:**
```json
{
  "explanation": "Based on the chest X-ray findings..."
}
```

## 🧪 Testing

### Testing Single Prediction with cURL

```bash
curl -X POST "http://localhost:8080/predict" \
  -F "file=@/path/to/xray.jpg"
```

### Testing Batch Prediction with cURL

```bash
curl -X POST "http://localhost:8080/predict_batch" \
  -F "files=@/path/to/xray1.jpg" \
  -F "files=@/path/to/xray2.jpg" \
  -F "files=@/path/to/xray3.jpg"
```

### Testing with Python Script

A test script is provided in `backend/test_batch_api.py`:

```bash
cd backend
python test_batch_api.py
```

This will automatically test the batch API with sample images from the `sample_uploads/` directory.

## ⚙️ Configuration

Στο `frontend/index.html`, άλλαξε το API URL αν χρειάζεται:

```javascript
// Line ~170
const API_URL = "http://localhost:8080/predict";
```

## 📊 Risk Classification Categories

The batch processing feature automatically classifies images into three risk levels based on the top predicted condition:

### 🔴 High Risk
- Consolidation
- Pneumothorax
- Edema
- Pneumonia
- Lung Lesion

### 🟡 Medium Risk
- Atelectasis
- Effusion
- Cardiomegaly
- Enlarged Cardiomediastinum
- Lung Opacity

### 🟢 Low Risk
- Fracture

**Note:** Conditions not listed above default to Medium Risk.

## 📊 Supported Pathologies

The model can detect the following pathologies from chest X-rays:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural Thickening
- Pneumonia
- Pneumothorax
- Lung Lesion
- Lung Opacity
- Enlarged Cardiomediastinum
- Fracture

## 🔧 Troubleshooting

### CORS Errors
Βεβαιώσου ότι το backend τρέχει και ότι το `API_URL` στο frontend είναι σωστό.

### Model Download
Την πρώτη φορά, το TorchXRayVision θα κατεβάσει το μοντέλο (~500MB). Περίμενε λίγα λεπτά.

### GPU Memory
Αν έχεις GPU memory issues, το μοντέλο θα πέσει αυτόματα σε CPU.

## 📝 Notes

- Το μοντέλο είναι για **research/demo purposes** μόνο
- Δεν πρέπει να χρησιμοποιηθεί για κλινικές αποφάσεις
- Τα X-rays πρέπει να είναι frontal chest radiographs

## 🎓 Credits

- Model: [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
- Dataset: [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- Grad-CAM: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

**Hackathon UOI 2025** 🚀