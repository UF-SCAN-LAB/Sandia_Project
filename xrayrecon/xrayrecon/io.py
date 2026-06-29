
import os
import glob
import numpy as np
from PIL import Image

def load_data(path: str) -> np.ndarray:
    """
    Load 3D data from a file or a directory of images.
    
    Args:
        path (str): Path to a .npy file or a directory containing images.
        
    Returns:
        np.ndarray: 3D numpy array of shape (Z, H, W).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    if os.path.isdir(path):
        return _load_from_directory(path)
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        # Fallback: try to load as single image
        try:
             img = np.array(Image.open(path))
             # Ensure 3D: (1, H, W) if it came out 2D
             if img.ndim == 2:
                 return img[np.newaxis, ...]
             return img
        except Exception:
            raise ValueError(f"Unsupported file format or directory structure: {path}")

def _load_from_directory(path: str) -> np.ndarray:
    # Look for common image extensions
    extensions = ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path, ext)))
        # Also try ALL CAPS
        files.extend(glob.glob(os.path.join(path, ext.upper())))
    
    # Sort files to ensure order (usually by filename numbering)
    files = sorted(list(set(files)))
    
    if not files:
        raise ValueError(f"No image files found in directory: {path}")
        
    print(f"Found {len(files)} images in {path}. Loading...")
    
    images = []
    for f in files:
        img_obj = Image.open(f)
        # Convert to grayscale if RGB? Usually reconstruction data is single channel.
        # Let's preserve original mode but if RGB convert to grayscale? 
        # The user said "convert it into the appropriate numpy array stack".
        # X-ray is typically density/intensity, so 'L' or 'F' (float).
        # Safe bet is to convert to numpy and see.
        data = np.array(img_obj)
        images.append(data)
        
    stack = np.stack(images, axis=0) # Z, H, W
    print(f"Loaded data stack with shape: {stack.shape}")
    return stack
