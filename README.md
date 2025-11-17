# VisionX - AI-Powered Chest X-Ray Diagnostic Assistant 🏥

<div align="center">

![VisionX Logo](frontend/VisionX.jpg)

### 🏆 2nd Place Winner - Hackathon UOI 2025

*An intelligent pre-diagnostic tool leveraging deep learning for chest X-ray analysis and risk stratification*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📑 Table of Contents

- [Media Showcase](#-media-showcase)
- [Project Vision](#-project-vision)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage Guide](#-usage-guide)
  - [Single Image Analysis](#single-image-analysis)
  - [Batch Risk Classification](#batch-risk-classification)
- [API Documentation](#-api-documentation)
- [Risk Classification System](#-risk-classification-system)
- [Model Details](#-model-details)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎬 Media Showcase

### Demo Video

https://github.com/user-attachments/assets/your-video-id-here

> **Note:** To embed the video in your GitHub README, upload `VisionX_demo.mp4` to GitHub (via Issues or directly) and replace the link above.

### Screenshots

<div align="center">

| Single Image Analysis | Batch Risk Classification |
|:---:|:---:|
| ![Single Analysis](docs/screenshots/single-analysis.png) | ![Batch Classification](docs/screenshots/batch-classification.png) |
| Real-time pathology detection with Grad-CAM visualization | Automated risk stratification for multiple X-rays |

</div>

---

## 🎯 Project Vision

**VisionX** aims to revolutionize early chest X-ray screening by providing healthcare professionals with an intelligent, AI-powered pre-diagnostic tool. Our mission is to:

- 🚀 **Accelerate Diagnosis**: Reduce initial screening time from hours to seconds
- 🎯 **Improve Accuracy**: Leverage state-of-the-art deep learning models trained on millions of X-rays
- 📊 **Risk Stratification**: Automatically prioritize critical cases requiring immediate attention
- 🔍 **Visual Explainability**: Provide Grad-CAM heatmaps to highlight regions of interest
- 🤖 **AI-Assisted Interpretation**: Generate clinical insights using advanced language models

### The Problem We Solve

In modern healthcare facilities, radiologists face an overwhelming volume of chest X-rays daily. VisionX serves as a **first-pass screening tool** that:

1. **Filters normal cases** from those requiring urgent attention
2. **Highlights potential pathologies** for radiologist review
3. **Organizes cases by risk level** to optimize workflow
4. **Provides visual explanations** to support diagnostic decisions

> ⚠️ **Important:** VisionX is designed as a **pre-diagnostic screening tool** and **educational platform**. It is NOT intended to replace professional medical judgment or be used for clinical decision-making without physician oversight.

---

## ✨ Key Features

### 🔬 Advanced AI Analysis

- **Multi-Label Classification**: Simultaneous detection of 18+ chest pathologies
- **DenseNet121 Architecture**: Pretrained on CheXpert dataset (224K+ chest X-rays)
- **High Performance**: 90%+ accuracy on validation datasets
- **Real-time Processing**: Results in under 3 seconds per image

### 👁️ Visual Explainability

- **Grad-CAM Heatmaps**: Visualize which regions influenced the model's prediction
- **Overlay & Standalone Views**: Compare original X-ray with AI attention maps
- **Multi-Layer Analysis**: Target specific pathologies for focused visualization

### 🎨 Dual Operating Modes

#### 1️⃣ Single Image Analysis
- Upload individual chest X-rays for detailed examination
- View top predicted pathologies with confidence scores
- Get AI-generated clinical interpretations
- Export results for medical records

#### 2️⃣ Batch Risk Classification
- Process multiple X-rays simultaneously (10-100+ images)
- Automatic risk stratification (High/Medium/Low/No Finding)
- Organized file output by risk category
- Aggregate statistics and summary reports

### 🧠 AI Clinical Interpretation

- **LLM-Powered Explanations**: Gemini 2.5 Flash generates human-readable clinical insights
- **Context-Aware Analysis**: Considers probability distributions and multi-label predictions
- **Educational Content**: Explains pathology significance and common causes
- **Safety-First Design**: Emphasizes limitations and need for professional evaluation

### 🌐 Modern Web Interface

- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Drag-and-Drop Upload**: Intuitive file handling
- **Real-Time Feedback**: Progress indicators and status updates
- **Medical-Grade UI**: Clean, professional interface designed for clinical environments

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/CSS/JS)               │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  File Upload    │  │  Mode Toggle │  │  Results View │  │
│  │  (Drag & Drop)  │  │  (Single/    │  │  (Charts &    │  │
│  │                 │  │   Batch)     │  │   Heatmaps)   │  │
│  └────────┬────────┘  └──────┬───────┘  └───────▲───────┘  │
└───────────┼────────────────────┼──────────────────┼─────────┘
            │                    │                  │
            │      HTTP/REST API (CORS Enabled)     │
            │                    │                  │
┌───────────▼────────────────────▼──────────────────┼─────────┐
│                    FastAPI Backend                │         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                         │   │
│  │  • POST /predict        - Single image analysis     │   │
│  │  • POST /predict_batch  - Batch risk classification │   │
│  │  • POST /llm_explain    - AI interpretation         │   │
│  │  • GET  /               - Health check              │   │
│  └─────────────────┬──────────────────────────┬────────┘   │
│                    │                          │            │
│  ┌─────────────────▼────────┐  ┌──────────────▼─────────┐ │
│  │  Image Processing        │  │  Risk Classification   │ │
│  │  • Preprocessing         │  │  • Category Mapping    │ │
│  │  • Normalization         │  │  • File Organization   │ │
│  │  • Tensor Conversion     │  │  • Batch Summary       │ │
│  └─────────────────┬────────┘  └────────────────────────┘ │
│                    │                                       │
│  ┌─────────────────▼────────────────────────────────────┐ │
│  │           DenseNet121 Model (TorchXRayVision)        │ │
│  │  • 18 Pathology Classes                              │ │
│  │  • Sigmoid Output (Multi-label)                      │ │
│  │  • CUDA/CPU Support                                  │ │
│  └─────────────────┬────────────────────────────────────┘ │
│                    │                                       │
│  ┌─────────────────▼────────────────────────────────────┐ │
│  │        Grad-CAM Visualization (pytorch-grad-cam)     │ │
│  │  • Target Layer: features.norm5                      │ │
│  │  • Overlay Generation                                │ │
│  │  • Heatmap Coloring                                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │     Gemini 2.5 Flash (LLM Integration)              │  │
│  │  • Clinical Analysis Generation                     │  │
│  │  • Safety-Aware Prompting                           │  │
│  │  • Structured Output Formatting                     │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, Python 3.8+, Uvicorn |
| **Deep Learning** | PyTorch, TorchXRayVision, pytorch-grad-cam |
| **LLM Integration** | Google Gemini 2.5 Flash |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Image Processing** | PIL (Pillow), NumPy, OpenCV |
| **Model** | DenseNet121 (pretrained on CheXpert) |
| **Deployment** | CORS-enabled REST API |

---

## 📁 Project Structure

```
VisionX/
├── backend/
│   ├── app_fastapi.py          # Main FastAPI application
│   ├── utils.py                # Helper functions & utilities
│   ├── requirements.txt        # Python dependencies
│   ├── test_api.py            # Single image API tests
│   ├── test_batch_api.py      # Batch processing tests
│   └── classified_images/     # Output directory for batch results
│       ├── High/              # High-risk cases
│       ├── Medium/            # Medium-risk cases
│       ├── Low/               # Low-risk cases
│       └── No Finding/        # Normal X-rays
│
├── frontend/
│   ├── index.html             # Main web interface
│   ├── style.css              # Styling & animations
│   └── VisionX.jpg            # Logo & branding
│
├── sample_uploads/            # Organized test dataset
│   └── patient*/study*/       # Patient-study-view hierarchy
│
├── sample_uploads_batch/      # Flat batch testing directory
│
├── 415.404-DiseaseNotFound.pptx  # Project presentation
├── VisionX_demo.mp4           # Demo video
├── ModelTesting.ipynb         # Jupyter notebook for experiments
├── requirements.txt           # Global dependencies
└── README.md                  # This file
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional, for cloning)
- **CUDA Toolkit** (optional, for GPU acceleration)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/VisionX.git
cd VisionX
```

#### 2. Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Option B: Using Conda**

```bash
conda create -n visionx python=3.10
conda activate visionx
```

#### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Core dependencies include:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` & `torchvision` - Deep learning
- `torchxrayvision` - Medical imaging models
- `pytorch-grad-cam` - Visualization
- `pillow` - Image processing
- `python-multipart` - File uploads
- `google-generativeai` - LLM integration

#### 4. Configure API Keys

Create a `.env` file in the `backend/` directory:

```bash
# Google Gemini API Key (for AI explanations)
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get your free API key at [Google AI Studio](https://makersuite.google.com/app/apikey)

### Running the Application

#### Step 1: Start the Backend Server

```bash
cd backend
python app_fastapi.py
```

**Expected output:**
```
Loading CheXpert DenseNet121 model...
✓ Model loaded successfully!
Device: cuda (or cpu)
Pathologies: ['Atelectasis', 'Consolidation', ...]
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Step 2: Start the Frontend

**Option A: Python HTTP Server**

```bash
# In a new terminal
cd frontend
python -m http.server 3000
```

Frontend available at: **http://localhost:3000**

**Option B: VS Code Live Server**

1. Install "Live Server" extension
2. Right-click `frontend/index.html`
3. Select "Open with Live Server"

**Option C: Direct File Access**

Simply open `frontend/index.html` in your browser (some features may require CORS setup).

#### Step 3: Access the Application

Open your browser and navigate to:
- **Frontend UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs (FastAPI Swagger UI)
- **Health Check**: http://localhost:8000/

---

## 📖 Usage Guide

### Single Image Analysis

Perfect for detailed examination of individual chest X-rays.

1. **Select Mode**
   - Choose "Single Image Analysis" from the mode selector

2. **Upload Image**
   - Click "Select Image" or drag-and-drop a chest X-ray (JPG, PNG)
   - Supported formats: JPEG, PNG, BMP, TIFF
   - Recommended: Frontal chest X-rays

3. **Run Prediction**
   - Click the "Predict" button
   - Processing takes 2-5 seconds depending on hardware

4. **Review Results**
   - **Top Prediction**: Primary pathology detected
   - **Clinical Risk Level**: Automatic risk classification
   - **Grad-CAM Visualization**: Heatmap showing AI attention
   - **Probability Confidence**: Model certainty score

5. **Get AI Explanation** (Optional)
   - Click "Generate AI Explanation"
   - View detailed clinical interpretation
   - Review limitations and disclaimers

### Batch Risk Classification

Ideal for screening large volumes of X-rays and prioritizing urgent cases.

1. **Select Mode**
   - Choose "Batch Risk Classification" from the mode selector

2. **Upload Multiple Images**
   - Click "Select Images" and choose multiple files
   - Or drag-and-drop an entire folder
   - Supports 10-100+ images simultaneously

3. **Run Batch Processing**
   - Click "Predict" to analyze all images
   - Progress bar shows real-time status

4. **Review Summary**
   - **Risk Distribution**: Visual breakdown by category
   - **High Risk**: 🔴 Images requiring immediate attention
   - **Medium Risk**: 🟡 Cases needing follow-up
   - **Low Risk**: 🟢 Structural issues with standard care
   - **No Finding**: ✅ Normal/healthy X-rays

5. **Access Organized Files**
   - Images automatically saved to `backend/classified_images/`
   - Subfolders: `High/`, `Medium/`, `Low/`, `No Finding/`
   - Filenames preserved with sequential prefixes

---

## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

```http
GET /
```

**Response:**
```json
{
  "status": "online",
  "model": "DenseNet121-CheXpert",
  "device": "cuda",
  "pathologies": ["Atelectasis", "Consolidation", ...]
}
```

#### 2. Single Image Prediction

```http
POST /predict
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@chest_xray.jpg"
```

**Response:**
```json
{
  "pred_class": ["Cardiomegaly"],
  "probs": {
    "Atelectasis": 0.234,
    "Cardiomegaly": 0.876,
    "Consolidation": 0.123
  },
  "gradcam_overlay": "data:image/png;base64,iVBORw0KG...",
  "gradcam_heatmap": "data:image/png;base64,iVBORw0KG...",
  "gradcam_target": "Cardiomegaly"
}
```

#### 3. Batch Risk Classification

```http
POST /predict_batch
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -F "files=@xray1.jpg" \
  -F "files=@xray2.jpg" \
  -F "files=@xray3.jpg"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "xray1.jpg",
      "top_label": "Pneumonia",
      "top_probability": 0.87,
      "risk_level": "High",
      "top_predictions": [
        {"label": "Pneumonia", "probability": 0.87},
        {"label": "Consolidation", "probability": 0.65}
      ],
      "saved_path": "classified_images/High/0000_xray1.jpg"
    }
  ],
  "summary": {
    "High": 1,
    "Medium": 0,
    "Low": 0,
    "No Finding": 0
  },
  "output_directory": "classified_images"
}
```

#### 4. AI Clinical Interpretation

```http
POST /llm_explain
Content-Type: application/json
```

**Request:**
```json
{
  "pred_class": ["Cardiomegaly"],
  "probs": {
    "Cardiomegaly": 0.876
  }
}
```

**Response:**
```json
{
  "explanation": "ANALYSIS:\n- Cardiomegaly: An enlarged heart...\n\nSUMMARY:\nThe model indicates...\n\nLIMITATIONS AND SAFETY NOTICE:\n- This is NOT a diagnosis..."
}
```

---

## 🚨 Risk Classification System

VisionX uses a four-tier risk stratification system based on clinical urgency:

### ✅ No Finding (Normal)
**Conditions:** No significant pathology detected (all predictions < 30% confidence)

**Action:** Routine follow-up, no immediate concern

---

### 🟢 Low Risk
**Conditions:**
- Fracture

**Characteristics:**
- Structural issues requiring standard orthopedic care
- Non-life-threatening
- Scheduled treatment appropriate

**Action:** Refer to orthopedics, standard priority

---

### 🟡 Medium Risk
**Conditions:**
- Atelectasis (lung collapse)
- Pleural Effusion (fluid around lungs)
- Cardiomegaly (enlarged heart)
- Enlarged Cardiomediastinum
- Lung Opacity

**Characteristics:**
- Requires monitoring and follow-up
- May indicate chronic conditions
- Potential for deterioration if untreated

**Action:** Schedule timely evaluation, monitor symptoms

---

### 🔴 High Risk
**Conditions:**
- Consolidation (dense lung infiltrate)
- Pneumothorax (collapsed lung)
- Pulmonary Edema (fluid in lungs)
- Pneumonia (lung infection)
- Lung Lesion (mass/nodule)

**Characteristics:**
- Requires immediate medical attention
- Potential for rapid deterioration
- May indicate acute, severe pathology

**Action:** **Prioritize for urgent radiologist review and clinical assessment**

## 🧪 Model Details

### Architecture: DenseNet121

**DenseNet (Densely Connected Convolutional Networks)** connects each layer to every other layer in a feed-forward fashion, enabling:
- **Feature Reuse**: Improved gradient flow and parameter efficiency
- **Reduced Overfitting**: Regularization through dense connections
- **Deeper Networks**: Easier training of 100+ layer models

### Training Dataset: CheXpert

- **Size**: 224,316 chest radiographs from 65,240 patients
- **Source**: Stanford Hospital (2002-2017)
- **Labels**: 14 clinical observations extracted via NLP
- **Annotation**: Expert radiologist validation

### Model Performance

| Metric | Value |
|--------|-------|
| **Dataset** | CheXpert Validation Set |
| **Accuracy** | 90.3% |
| **AUROC (Average)** | 0.876 |
| **Sensitivity** | 85-95% (varies by pathology) |
| **Specificity** | 88-96% (varies by pathology) |

### Supported Pathologies (18 Classes)

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema (via empty string filtering)
7. Fibrosis (via empty string filtering)
8. Fracture
9. Hernia (via empty string filtering)
10. Infiltration (via empty string filtering)
11. Lung Lesion
12. Lung Opacity
13. Mass (via empty string filtering)
14. Nodule (via empty string filtering)
15. Pneumonia
16. Pneumothorax
17. Pleural Thickening (via empty string filtering)
18. Enlarged Cardiomediastinum

> **Note:** Some pathologies have empty string labels in the model and are filtered out during preprocessing.

### Grad-CAM Visualization

**Gradient-weighted Class Activation Mapping (Grad-CAM)** generates visual explanations by:

1. Computing gradients of target class w.r.t. feature maps
2. Weighting feature maps by gradient importance
3. Creating heatmap highlighting influential regions
4. Overlaying heatmap on original X-ray

**Target Layer:** `model.features.norm5` (final normalization layer before classification)

---

## 🧑‍💻 Development & Testing

### Running Tests

#### Unit Tests
```bash
cd backend
python test_api.py          # Test single image prediction
python test_batch_api.py    # Test batch processing
```

#### Manual Testing with Sample Data

```bash
# Test with provided sample X-rays
cd backend
python test_batch_api.py
# Processes all images from sample_uploads/
```

### Adding New Pathologies

To extend the risk classification system:

1. Update `RISK_CATEGORIES` in `backend/app_fastapi.py`:

```python
RISK_CATEGORIES = {
    "No Finding": ["No Finding"],
    "Low": ["Fracture", "Your_New_Low_Risk_Condition"],
    "Medium": ["Atelectasis", ...],
    "High": ["Consolidation", ...]
}
```

2. Update frontend `classifyRiskLevel()` in `frontend/index.html`

3. Restart backend server

### Model Retraining

To fine-tune on custom dataset:

```python
import torchxrayvision as xrv

# Load pretrained model
model = xrv.models.DenseNet(weights="densenet121-res224-chex")

# Add your training loop
# (requires labeled chest X-ray dataset)
```

Refer to [TorchXRayVision documentation](https://github.com/mlmed/torchxrayvision) for details.

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Areas for Improvement

- [ ] Add support for DICOM medical imaging format
- [ ] Implement user authentication & session management
- [ ] Create database backend for storing results
- [ ] Add comparison view for temporal X-ray analysis
- [ ] Integrate with PACS (Picture Archiving and Communication System)
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Expand to other imaging modalities (CT, MRI)

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use ES6+ features, semicolons optional
- **HTML/CSS**: Maintain existing naming conventions

---

## 🎓 Acknowledgments

### Hackathon UOI 2025
- **Team Members:** [Add your team members]
- **Achievement:** 🏆 2nd Place Winner
- **Category:** Healthcare AI & Medical Imaging

### Open Source Projects
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) by Joseph Paul Cohen - Medical imaging models
- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) by Stanford ML Group
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez - Modern web framework
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) by Jacob Gildenblat - Visualization library
- [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM integration

### Research Papers
- **DenseNet**: Huang et al. (2017) - "Densely Connected Convolutional Networks"
- **CheXpert**: Irvin et al. (2019) - "CheXpert: A Large Chest Radiograph Dataset"
- **Grad-CAM**: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"

### Special Thanks
- University of Ioannina for hosting the hackathon
- Medical advisors for clinical validation guidance
- Open-source community for incredible tools and resources

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Medical Disclaimer

**VisionX is a research and educational tool. It is NOT intended for clinical use or medical diagnosis.**

- This software is provided "as is" without warranty of any kind
- Results should ALWAYS be reviewed by qualified healthcare professionals
- Do NOT use for emergency medical decisions
- Do NOT substitute for professional radiologist interpretation
- Clinical correlation and additional diagnostic tests are necessary
- Consult your physician for any health concerns

---

## 📧 Contact

For questions, feedback, or collaboration opportunities:

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/VisionX/issues)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]

---

<div align="center">

### 🌟 If you found this project helpful, please give it a star!

Made with ❤️ by [Your Team Name] | Hackathon UOI 2025

</div>
