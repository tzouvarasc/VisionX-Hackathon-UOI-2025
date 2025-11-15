"""
Test script for batch image classification API
"""
import requests
import os
from pathlib import Path

API_URL = "http://localhost:8080/predict_batch"

def test_batch_predict():
    """Test batch prediction with sample images"""
    
    # Get sample images
    sample_dir = Path("../sample_uploads")
    image_files = []
    
    # Collect some sample images
    for patient_dir in list(sample_dir.glob("patient*"))[:5]:  # Take first 5 patients
        study_dir = patient_dir / "study1"
        if study_dir.exists():
            for img_file in study_dir.glob("*.jpg"):
                image_files.append(img_file)
                if len(image_files) >= 10:  # Test with 10 images
                    break
        if len(image_files) >= 10:
            break
    
    if not image_files:
        print("❌ No sample images found!")
        return
    
    print(f"📁 Found {len(image_files)} sample images")
    print(f"Images: {[f.name for f in image_files]}")
    
    # Prepare multipart form data
    files = []
    for img_path in image_files:
        files.append(('files', (img_path.name, open(img_path, 'rb'), 'image/jpeg')))
    
    print(f"\n🚀 Sending batch request to {API_URL}...")
    
    try:
        response = requests.post(API_URL, files=files)
        
        # Close file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Batch prediction successful!")
            print(f"\n📊 Summary:")
            print(f"  🔴 High Risk: {data['summary']['High']}")
            print(f"  🟡 Medium Risk: {data['summary']['Medium']}")
            print(f"  🟢 Low Risk: {data['summary']['Low']}")
            print(f"\n📂 Images saved to: {data['output_directory']}")
            
            print(f"\n📋 Individual Results:")
            for result in data['results']:
                if 'error' in result:
                    print(f"  ❌ {result['filename']}: {result['error']}")
                else:
                    print(f"  {result['filename']}:")
                    print(f"    - Risk: {result['risk_level']}")
                    print(f"    - Top prediction: {result['top_label']} ({result['top_probability']*100:.1f}%)")
                    print(f"    - Saved to: {result['saved_path']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Batch Image Classification Test")
    print("=" * 60)
    test_batch_predict()

