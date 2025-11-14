"""
Test script for CheXpert FastAPI backend
Usage: python test_api.py <path_to_xray_image>
"""

import requests
import sys
import json
from pathlib import Path

API_URL = "http://localhost:8080"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        response.raise_for_status()
        data = response.json()
        print("✅ Health check passed!")
        print(f"   Status: {data['status']}")
        print(f"   Model: {data['model']}")
        print(f"   Device: {data['device']}")
        print(f"   Pathologies: {len(data['pathologies'])} classes")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_prediction(image_path):
    """Test the prediction endpoint"""
    print(f"\n📸 Testing prediction with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/predict", files=files)
            response.raise_for_status()
        
        data = response.json()
        
        print("✅ Prediction successful!")
        print(f"\n🏆 Top 5 Predictions:")
        for i, pred in enumerate(data['pred_class'][:5], 1):
            prob = data['probs'][pred]
            print(f"   {i}. {pred:20s} {prob:.3f} ({prob*100:.1f}%)")
        
        print(f"\n🔥 Grad-CAM Target: {data['gradcam_target']}")
        
        # Check if Grad-CAM images are returned
        if data.get('gradcam_overlay') and data.get('gradcam_heatmap'):
            print(f"   ✓ Grad-CAM overlay: {len(data['gradcam_overlay'])} chars")
            print(f"   ✓ Grad-CAM heatmap: {len(data['gradcam_heatmap'])} chars")
        
        # Save full response to JSON
        output_file = "test_prediction_result.json"
        # Remove base64 images for readable JSON
        data_copy = data.copy()
        if 'gradcam_overlay' in data_copy:
            data_copy['gradcam_overlay'] = f"<base64 data: {len(data['gradcam_overlay'])} chars>"
        if 'gradcam_heatmap' in data_copy:
            data_copy['gradcam_heatmap'] = f"<base64 data: {len(data['gradcam_heatmap'])} chars>"
        
        with open(output_file, 'w') as f:
            json.dump(data_copy, f, indent=2)
        print(f"\n💾 Full results saved to: {output_file}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("=" * 60)
    print("CheXpert API Test Script")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        print("\n⚠️  Backend not running! Start it with:")
        print("   python app_fastapi.py")
        sys.exit(1)
    
    # Test prediction if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_prediction(image_path)
    else:
        print("\n💡 To test prediction, run:")
        print("   python test_api.py <path_to_xray_image>")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()