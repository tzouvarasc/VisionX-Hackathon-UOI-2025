# Batch Processing & Risk Classification Guide 📊

## Overview

The batch processing feature allows you to upload multiple chest X-ray images at once and automatically classify them into risk categories based on the AI model's predictions.

## How It Works

1. **Upload**: Select multiple X-ray images (any number)
2. **Process**: The model analyzes each image and predicts conditions
3. **Classify**: Based on the top predicted condition, each image is assigned a risk level
4. **Organize**: Images are automatically saved into folders by risk category

## Risk Classification Logic

The risk level is determined by the **top predicted condition** (highest probability):

### 🔴 High Risk
**Conditions that require immediate attention:**
- **Consolidation**: Lung tissue filled with fluid/material
- **Pneumothorax**: Collapsed lung
- **Edema**: Fluid buildup in lungs
- **Pneumonia**: Lung infection
- **Lung Lesion**: Abnormal tissue in lungs

### 🟡 Medium Risk
**Conditions that need monitoring:**
- **Atelectasis**: Partial lung collapse
- **Effusion**: Fluid around lungs
- **Cardiomegaly**: Enlarged heart
- **Enlarged Cardiomediastinum**: Widened chest center
- **Lung Opacity**: Unclear lung areas

### 🟢 Low Risk
**Structural issues:**
- **Fracture**: Bone breaks

## Using the Web Interface

### Step-by-Step Guide

1. **Select Batch Mode**
   - Open the web application
   - Click "Batch Risk Classification" radio button

2. **Upload Images**
   - Click "Select Image" or drag & drop multiple files
   - Supported formats: JPG, PNG, JPEG
   - No limit on number of images

3. **Run Classification**
   - Click "Predict" button
   - Wait for processing (progress bar shows status)

4. **View Results**
   - Summary shows count of images in each risk category
   - Scroll to see individual results grouped by risk level
   - Each result shows:
     - Filename
     - Top predicted condition
     - Confidence percentage
     - Top 3 predictions

5. **Access Saved Images**
   - Images are saved to `classified_images/` folder
   - Organized into `High/`, `Medium/`, `Low/` subfolders
   - Original filenames preserved with index prefix

## Using the API

### Batch Prediction Endpoint

**URL:** `POST http://localhost:8080/predict_batch`

**Request:**
```bash
curl -X POST "http://localhost:8080/predict_batch" \
  -F "files=@patient1_xray.jpg" \
  -F "files=@patient2_xray.jpg" \
  -F "files=@patient3_xray.jpg"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "patient1_xray.jpg",
      "index": 0,
      "top_label": "Pneumonia",
      "top_probability": 0.87,
      "risk_level": "High",
      "top_predictions": [
        {"label": "Pneumonia", "prob": 0.87},
        {"label": "Consolidation", "prob": 0.65},
        {"label": "Infiltration", "prob": 0.43}
      ],
      "all_probs": {
        "Atelectasis": 0.234,
        "Cardiomegaly": 0.123,
        ...
      },
      "saved_path": "classified_images/High/0000_patient1_xray.jpg"
    },
    ...
  ],
  "summary": {
    "High": 5,
    "Medium": 12,
    "Low": 3
  },
  "output_directory": "classified_images"
}
```

## Python Example

```python
import requests
from pathlib import Path

API_URL = "http://localhost:8080/predict_batch"

# Collect image files
image_paths = list(Path("xrays/").glob("*.jpg"))

# Prepare multipart form data
files = [('files', (img.name, open(img, 'rb'), 'image/jpeg')) 
         for img in image_paths]

# Send request
response = requests.post(API_URL, files=files)

# Close file handles
for _, (_, file_obj, _) in files:
    file_obj.close()

# Process results
if response.status_code == 200:
    data = response.json()
    print(f"High Risk: {data['summary']['High']}")
    print(f"Medium Risk: {data['summary']['Medium']}")
    print(f"Low Risk: {data['summary']['Low']}")
    
    for result in data['results']:
        print(f"{result['filename']}: {result['risk_level']} - {result['top_label']}")
```

## Output Structure

```
classified_images/
├── High/
│   ├── 0000_patient1_xray.jpg
│   ├── 0003_patient4_xray.jpg
│   └── 0007_patient8_xray.jpg
├── Medium/
│   ├── 0001_patient2_xray.jpg
│   ├── 0004_patient5_xray.jpg
│   └── 0005_patient6_xray.jpg
└── Low/
    └── 0002_patient3_xray.jpg
```

## Best Practices

### Image Quality
- Use frontal chest X-rays for best results
- Ensure images are clear and properly exposed
- DICOM files should be converted to JPG/PNG first

### Batch Size
- No hard limit on number of images
- For large batches (100+), consider splitting into multiple requests
- Processing time depends on hardware (GPU vs CPU)

### Interpreting Results
- Risk classification is based on AI predictions only
- Always review individual predictions and probabilities
- Use as a triage tool, not for diagnosis
- Clinical correlation is essential

## Use Cases

### 1. Emergency Department Triage
- Quickly identify high-risk cases needing immediate attention
- Prioritize radiologist review based on risk level

### 2. Research Studies
- Batch process research datasets
- Organize images by predicted conditions
- Generate statistical summaries

### 3. Quality Control
- Flag unusual findings for review
- Monitor distribution of conditions in patient population

### 4. Archive Organization
- Automatically categorize historical X-rays
- Facilitate retrospective studies

## Limitations

⚠️ **Important Disclaimers:**

1. **Not for Clinical Use**: This is a research/demo tool only
2. **AI Limitations**: Model may make errors or miss conditions
3. **Requires Validation**: All results need expert review
4. **Single View**: Only analyzes one image at a time (no multi-view correlation)
5. **Training Data Bias**: Performance may vary based on image quality and patient demographics

## Troubleshooting

### Images Not Processing
- Check image format (JPG, PNG supported)
- Verify images are chest X-rays (frontal view works best)
- Ensure backend server is running

### Incorrect Risk Classification
- Review the top predicted condition
- Check probability scores (low confidence = uncertain)
- Verify risk category mappings match your use case

### Performance Issues
- Large batches may take time (especially on CPU)
- Consider processing in smaller batches
- GPU acceleration recommended for production use

### Output Folder Issues
- Ensure write permissions for `classified_images/` directory
- Check disk space availability
- Folder is created automatically if it doesn't exist

## Technical Details

### Model
- **Architecture**: DenseNet121
- **Training**: CheXpert dataset
- **Input**: 224x224 grayscale images
- **Output**: 18 pathology probabilities

### Classification Algorithm
1. Process each image through DenseNet
2. Apply sigmoid to get probabilities
3. Select condition with highest probability
4. Map condition to risk category
5. Save image to appropriate folder

### Performance
- **Single Image**: ~100-500ms (GPU) / ~1-3s (CPU)
- **Batch of 10**: ~1-5s (GPU) / ~10-30s (CPU)
- Scales linearly with batch size

## Future Enhancements

Potential improvements for future versions:

- [ ] Configurable risk categories
- [ ] Custom classification rules
- [ ] Multi-view correlation (frontal + lateral)
- [ ] Confidence thresholds
- [ ] Export results to CSV/Excel
- [ ] Integration with PACS systems
- [ ] Detailed PDF reports per patient

---

**Questions or Issues?** Contact the development team or open an issue on GitHub.

**Hackathon UOI 2025** 🚀

