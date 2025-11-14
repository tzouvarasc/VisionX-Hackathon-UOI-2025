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

1. **Upload X-ray**: Drag & drop ή click "Select Image"
2. **Predict**: Click "Predict" button
3. **View Results**: 
   - Top 5 pathology predictions
   - Probability bars
   - Grad-CAM heatmaps (overlay & pure heatmap)

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
Upload chest X-ray for prediction

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

## 🧪 Testing with cURL

```bash
curl -X POST "http://localhost:8080/predict" \
  -F "file=@/path/to/xray.jpg"
```

## ⚙️ Configuration

Στο `frontend/index.html`, άλλαξε το API URL αν χρειάζεται:

```javascript
// Line ~170
const API_URL = "http://localhost:8080/predict";
```

## 📊 Supported Pathologies

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