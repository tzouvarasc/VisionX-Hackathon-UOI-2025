"""
Script to copy all images from sample_uploads (recursively) to sample_uploads_batch folder
"""
import os
import shutil
from pathlib import Path

def copy_images_to_batch():
    """
    Recursively find all images in sample_uploads and copy them to sample_uploads_batch
    """
    # Define source and destination folders
    source_folder = Path("sample_uploads")
    dest_folder = Path("sample_uploads_batch")
    
    # Create destination folder if it doesn't exist
    dest_folder.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    
    # Counter for copied files
    copied_count = 0
    
    print(f"Searching for images in: {source_folder.absolute()}")
    print(f"Destination folder: {dest_folder.absolute()}")
    print("-" * 60)
    
    # Recursively find all image files
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # Check if file has an image extension
            file_ext = Path(filename).suffix.lower()
            if file_ext in image_extensions:
                # Get source file path
                source_path = Path(root) / filename
                
                # Create a unique destination filename
                # Use the patient/study folder names to make unique names
                relative_path = source_path.relative_to(source_folder)
                # Replace path separators with underscores for flat structure
                dest_filename = str(relative_path).replace(os.sep, '_')
                dest_path = dest_folder / dest_filename
                
                # If file already exists, add a counter
                counter = 1
                original_dest_path = dest_path
                while dest_path.exists():
                    stem = original_dest_path.stem
                    suffix = original_dest_path.suffix
                    dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Copy the file
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    print(f"[OK] Copied: {source_path} -> {dest_path.name}")
                except Exception as e:
                    print(f"[ERROR] Error copying {source_path}: {e}")
    
    print("-" * 60)
    print(f"[SUCCESS] Successfully copied {copied_count} images to {dest_folder}")
    print(f"[INFO] Total files in destination: {len(list(dest_folder.glob('*')))}")

if __name__ == "__main__":
    print("=" * 60)
    print("Image Copy Script - sample_uploads to sample_uploads_batch")
    print("=" * 60)
    print()
    
    copy_images_to_batch()
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

