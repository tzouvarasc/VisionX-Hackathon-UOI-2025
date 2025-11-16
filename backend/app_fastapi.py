from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
import numpy as np
from PIL import Image
import io
import base64
from utils import encode_rgb_to_base64, preprocess_image, create_heatmap_images

import os
from typing import Dict, List, Union, Optional
import shutil
from pathlib import Path

from pydantic import BaseModel
from google import genai  # Gemini SDK

# The client picks up the API key from GEMINI_API_KEY env var
GEMINI_API_KEY = "AIzaSyDoQHu5eYIzmmJL9tdAsyL_ODXEP7Ceulc"
if not GEMINI_API_KEY:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI(
    title="CheXpert DenseNet API",
    description="Chest X-ray classification with Grad-CAM visualizations",
    version="1.0.0"
)

# CORS configuration - επιτρέπει requests από το frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Σε production βάλε το specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None

# Risk classification mapping
RISK_CATEGORIES = {
    "No Finding": ["No Finding"],  # Normal/healthy chest X-ray
    "Low": ["Fracture"],
    "Medium": ["Atelectasis", "Effusion", "Cardiomegaly", "Enlarged Cardiomediastinum", "Lung Opacity"],
    "High": ["Consolidation", "Pneumothorax", "Edema", "Pneumonia", "Lung Lesion"]
}

def classify_risk(top_label: str) -> str:
    """Classify image risk based on top predicted label"""
    for risk_level, conditions in RISK_CATEGORIES.items():
        if top_label in conditions:
            return risk_level
    return "Medium"  # Default to Medium if not found

@app.on_event("startup")
async def load_model():
    """Load the DenseNet model on startup"""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Pathologies: {model.pathologies}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "DenseNet121-CheXpert",
        "device": str(device),
        "pathologies": model.pathologies if model else []
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict chest X-ray pathologies and generate Grad-CAM visualizations
    
    Returns:
        - pred_class: Top prediction(s)
        - probs: Dictionary of all probabilities
        - gradcam_overlay: Base64 encoded overlay image
        - gradcam_heatmap: Base64 encoded heatmap
        - gradcam_target: Target class for Grad-CAM
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        img_tensor, img_rgb = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Create probabilities dictionary - filter out empty labels
        probs = {
            pathology: float(predictions[i])
            for i, pathology in enumerate(model.pathologies)
            if pathology and pathology.strip()  # Only include non-empty labels
        }
        
        # Get top predictions - only from non-empty labels
        valid_indices = [i for i, p in enumerate(model.pathologies) if p and p.strip()]
        valid_predictions = [(i, predictions[i]) for i in valid_indices]
        valid_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Check if this is a "No Finding" case (all predictions are low)
        max_prob = valid_predictions[0][1] if valid_predictions else 0.0
        
        if max_prob < 0.3:  # Threshold for "No Finding"
            top_predictions = ["No Finding"]
            top_class_name = "No Finding"
            # Add "No Finding" to probs with high confidence
            probs["No Finding"] = float(1.0 - max_prob)
            # Use a middle layer for visualization when no finding
            top_class_idx = valid_predictions[0][0] if valid_predictions else 0
        else:
            # Get top 5 valid predictions
            top_indices = [idx for idx, _ in valid_predictions[:5]]
            top_predictions = [model.pathologies[i] for i in top_indices]
            top_class_idx = top_indices[0]
            top_class_name = model.pathologies[top_class_idx]
        
        # Create Grad-CAM
        target_layers = [model.features.norm5]
        
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, class_idx):
                super().__init__()
                self.model = model
                self.class_idx = class_idx
            
            def forward(self, x):
                output = self.model(x)
                return output[:, self.class_idx:self.class_idx+1]
        
        wrapped_model = ModelWrapper(model, top_class_idx)
        wrapped_model.eval()
        
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
        
        # Create visualization images
        overlay_b64, heatmap_b64 = create_heatmap_images(img_rgb, grayscale_cam)

        processed_image_b64 = encode_rgb_to_base64(img_rgb)

        overall_confidence = float(np.mean(predictions))
        
        return JSONResponse(content={
            "pred_class": top_predictions,
            "probs": probs,
            "overall_confidence": overall_confidence,
            "processed_image": processed_image_b64,
            "gradcam_overlay": overlay_b64,
            "gradcam_heatmap": heatmap_b64,
            "gradcam_target": top_class_name
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Process multiple images, classify them by risk, and organize into folders
    
    Returns:
        - results: List of predictions for each image with risk classification
        - summary: Count of images per risk category
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create output directories
    output_base = Path("classified_images")
    output_base.mkdir(exist_ok=True)
    
    for risk_level in ["Low", "Medium", "High", "No Finding"]:
        (output_base / risk_level).mkdir(exist_ok=True)
    
    results = []
    risk_counts = {"Low": 0, "Medium": 0, "High": 0, "No Finding": 0}
    
    try:
        for idx, file in enumerate(files):
            try:
                # Read and validate image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert('RGB')
                
                # Preprocess image
                img_tensor, img_rgb = preprocess_image(image)
                img_tensor = img_tensor.to(device)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(img_tensor)
                    predictions = torch.sigmoid(outputs).cpu().numpy()[0]
                
                # Filter valid predictions (non-empty labels)
                valid_indices = [i for i, p in enumerate(model.pathologies) if p and p.strip()]
                valid_predictions = [(i, predictions[i]) for i in valid_indices]
                valid_predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Check if this is a "No Finding" case
                max_prob = valid_predictions[0][1] if valid_predictions else 0.0
                
                if max_prob < 0.3:  # Threshold for "No Finding"
                    top_label = "No Finding"
                    top_prob = 1.0 - max_prob  # Confidence in "No Finding"
                    risk_level = "No Finding"
                else:
                    top_idx = valid_predictions[0][0]
                    top_label = model.pathologies[top_idx]
                    top_prob = float(predictions[top_idx])
                    risk_level = classify_risk(top_label)
                
                # Update risk counts
                if risk_level not in risk_counts:
                    risk_counts[risk_level] = 0
                risk_counts[risk_level] += 1
                
                # Save image to appropriate folder
                original_name = file.filename or f"image_{idx}.jpg"
                safe_name = f"{idx:04d}_{original_name}"
                output_path = output_base / risk_level / safe_name
                
                # Create directory if it doesn't exist (for "No Finding")
                (output_base / risk_level).mkdir(exist_ok=True)
                
                # Save the original image
                image.save(output_path)
                
                # Create probabilities dictionary - filter out empty labels
                probs = {
                    pathology: float(predictions[i])
                    for i, pathology in enumerate(model.pathologies)
                    if pathology and pathology.strip()
                }
                
                # Get top 5 valid predictions
                top_predictions = [
                    {"label": model.pathologies[i], "prob": float(predictions[i])}
                    for i, _ in valid_predictions[:5]
                ]
                
                results.append({
                    "filename": original_name,
                    "index": idx,
                    "top_label": top_label,
                    "top_probability": top_prob,
                    "risk_level": risk_level,
                    "top_predictions": top_predictions,
                    "all_probs": probs,
                    "saved_path": str(output_path)
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename or f"image_{idx}",
                    "index": idx,
                    "error": str(e),
                    "risk_level": "Error"
                })
        
        return JSONResponse(content={
            "results": results,
            "summary": risk_counts,
            "output_directory": str(output_base)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    
class LLMExplainRequest(BaseModel):
    pred_class: Union[List[str], str]
    probs: Optional[Dict[str, float]] = None


@app.post("/llm_explain")
async def llm_explain(payload: LLMExplainRequest):
    """
    Use Gemini to describe what the illness/findings could be
    based on the model's output (classes + probabilities).

    This is strictly educational / research content,
    NOT a medical diagnosis.
    """

    # Normalize pred_class to list
    if isinstance(payload.pred_class, str):
        pred_classes = [payload.pred_class]
    else:
        pred_classes = payload.pred_class or []

    probs = payload.probs or {}

    # Build a compact text representation of the top 1 probability
    if probs:
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        if sorted_probs:
            top_name, top_p = sorted_probs[0]
            pct = round(float(top_p) * 100, 1)
            probs_text = f"- {top_name}: {pct}%"
        else:
            probs_text = "No probabilities available."
    else:
        probs_text = "No probabilities available."

    pred_classes_text = ", ".join(pred_classes) if pred_classes else "None"

    # Prompt for Gemini
    prompt = f"""
You are an experienced chest radiologist.

You are given the output of a chest X-ray multi-label classifier.
Based ONLY on the top predicted class and its probability, write a short
explanation of what this illness or radiographic finding might suggest.

IMPORTANT SAFETY REQUIREMENTS:
- Do NOT give medical advice, triage, or treatment recommendations.
- Do NOT tell the reader what they definitely "have".
- Emphasize that this is NOT a diagnosis and is for research/education only.
- Use clear, neutral language suitable for a medical student or junior doctor.
- If the probabilities are low, close together, or ambiguous, say that
  the findings are uncertain.
- You do NOT have access to the raw image or the clinical context.
- Keep the answer in plain text (no markdown formatting).
- Analyze ONLY the information given below.
- Each information will be analyzed in different paragraphs.
- Each paragraph will be in new line with a blank line in between.
- On the REFERENCE OUTPUT FORMAT section, the "//" indicates comments, do NOT include them in the final answer.
- The "[...]" indicates placeholders, replace them with the actual content.
- Uppercase between "[...]" indicates that the content should be replaced with the actual content.
- Follow the REFERENCE OUTPUT FORMAT strictly, DO NOT add ANY symbol or character not shown in the format (Even the invisible ones).

REFERENCE OUTPUT FORMAT:
// START OF OUTPUT //
ANALYSIS: // ON THIS STEP ONLY WRITE THE ANALYSIS OF THE TOP PREDICTED CLASS //
- [TOP PREDICTED CLASS] [detailed description of what this finding means, its clinical significance, and common causes]

SUMMARY: // A PARAGRAPH OF 3 TO 5 SENTENCES SUMMARIZING THE FINDINGS //
[final summary based on the ANALYSIS section above. Do NOT mention probabilities or model limitations here.]

LIMITATIONS AND SAFETY NOTICE: // A BULLETED LIST OF 3 POINTS REMINDING THE USER THAT: //
- This is NOT a diagnosis.
- This is for research and educational purposes only.
- Clinical correlation and further investigations are necessary.
// END OF OUTPUT //

Model output:
- Top predicted class: {pred_classes_text}

Class probability:
{probs_text}
""".strip()

    try:
        # Call Gemini (text-only)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",  # or another Gemini model you prefer
            contents=prompt,
        )
        explanation = (response.text or "").strip()
        if not explanation:
            raise RuntimeError("Empty explanation from Gemini")

    except Exception:
        # Safe fallback if Gemini call fails
        explanation = (
            "The AI explanation service is currently unavailable. "
            "You can still review the model's probability output to see which "
            "findings are more likely, but remember this tool is only for "
            "research and educational purposes and does not provide a diagnosis."
        )

    return {"explanation": explanation}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)