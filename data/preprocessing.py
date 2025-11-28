"""
Preprocessing for LoTSS Dataset
- Resize to 600x600
- Log-scale radio channel
- Normalize each channel separately
"""

import numpy as np
import cv2
from scipy.ndimage import zoom


class LoTSSPreprocessing:
    """
    Preprocessing for optical + radio images
    """
    
    def __init__(self, target_size=(600, 600)):
        self.target_size = target_size
    
    def __call__(self, optical, radio):
        """
        Args:
            optical: (H, W) numpy array
            radio: (H, W) numpy array
            
        Returns:
            optical_processed: (600, 600) normalized
            radio_processed: (600, 600) normalized with log-scaling
        """
        # Resize both to target size
        optical = self._resize(optical, self.target_size)
        radio = self._resize(radio, self.target_size)
        
        # Process optical
        optical = self._process_optical(optical)
        
        # Process radio (with log-scaling)
        radio = self._process_radio(radio)
        
        return optical, radio
    
    def _resize(self, img, target_size):
        """Resize image to target size"""
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        return img
    
    def _process_optical(self, optical):
        """
        Process optical channel:
        1. Clip extreme values
        2. Normalize to [0, 1]
        3. Standardize (mean=0, std=1)
        """
        # Handle NaN/Inf
        optical = np.nan_to_num(optical, nan=0.0, posinf=255.0, neginf=0.0)
        
        # Clip to valid range
        optical = np.clip(optical, 0, 255)
        
        # Normalize to [0, 1]
        if optical.max() > optical.min():
            optical = (optical - optical.min()) / (optical.max() - optical.min())
        else:
            optical = np.zeros_like(optical)
        
        # Standardize (mean=0, std=1)
        mean = optical.mean()
        std = optical.std() + 1e-8
        optical = (optical - mean) / std
        
        return optical.astype(np.float32)
    
    def _process_radio(self, radio):
        """
        Process radio channel:
        1. Handle sparse data
        2. Log-scale: log(1 + x)
        3. Normalize to [0, 1]
        4. Standardize
        """
        # Handle NaN/Inf
        radio = np.nan_to_num(radio, nan=0.0, posinf=255.0, neginf=0.0)
        
        # Clip to valid range
        radio = np.clip(radio, 0, 255)
        
        # Normalize to [0, 1] first
        if radio.max() > radio.min():
            radio = (radio - radio.min()) / (radio.max() - radio.min())
        else:
            radio = np.zeros_like(radio)
        
        # Log-scale to handle sparsity: log(1 + x)
        radio = np.log1p(radio)  # log(1 + x)
        
        # Standardize after log-scaling
        mean = radio.mean()
        std = radio.std() + 1e-8
        radio = (radio - mean) / std
        
        return radio.astype(np.float32)


if __name__ == "__main__":
    # Test preprocessing
    from PIL import Image
    from pathlib import Path
    
    # Load sample images
    data_root = Path("data/beautiful_dataset_v2/spiral_galaxies")
    optical_path = list((data_root / "optical").glob("*.png"))[0]
    radio_path = list((data_root / "radio_png").glob("*.png"))[0]
    
    optical = np.array(Image.open(optical_path).convert('L'))
    radio = np.array(Image.open(radio_path).convert('L'))
    
    print(f"Original shapes: optical={optical.shape}, radio={radio.shape}")
    print(f"Original ranges: optical=[{optical.min()}, {optical.max()}], radio=[{radio.min()}, {radio.max()}]")
    
    # Preprocess
    preprocess = LoTSSPreprocessing(target_size=(600, 600))
    optical_proc, radio_proc = preprocess(optical, radio)
    
    print(f"\nProcessed shapes: optical={optical_proc.shape}, radio={radio_proc.shape}")
    print(f"Processed ranges: optical=[{optical_proc.min():.4f}, {optical_proc.max():.4f}], radio=[{radio_proc.min():.4f}, {radio_proc.max():.4f}]")
    print(f"Processed means: optical={optical_proc.mean():.4f}, radio={radio_proc.mean():.4f}")
    print(f"Processed stds: optical={optical_proc.std():.4f}, radio={radio_proc.std():.4f}")
