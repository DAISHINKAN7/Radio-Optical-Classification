"""
Augmentation for LoTSS astronomical images
- Rotation (galaxies have no preferred orientation)
- Flipping
- Noise injection
- Brightness/contrast (channel-specific)
"""

import numpy as np
import cv2
from scipy.ndimage import rotate, gaussian_filter


class LoTSSAugmentation:
    """
    Augmentation for optical + radio images
    Applies consistent geometric transforms, channel-specific photometric
    """
    
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying augmentations
        """
        self.p = p
    
    def __call__(self, optical, radio):
        """
        Args:
            optical: (H, W) numpy array (already preprocessed)
            radio: (H, W) numpy array (already preprocessed)
            
        Returns:
            optical_aug, radio_aug: Augmented images
        """
        # Apply geometric transforms (same for both channels)
        if np.random.rand() < self.p:
            optical, radio = self._rotate(optical, radio)
        
        if np.random.rand() < self.p:
            optical, radio = self._flip(optical, radio)
        
        # Apply channel-specific photometric transforms
        if np.random.rand() < self.p:
            optical = self._adjust_optical(optical)
        
        if np.random.rand() < self.p:
            radio = self._adjust_radio(radio)
        
        # Add noise
        if np.random.rand() < self.p * 0.5:  # Lower probability for noise
            optical = self._add_noise(optical, scale=0.05)
            radio = self._add_noise(radio, scale=0.03)
        
        return optical, radio
    
    def _rotate(self, optical, radio):
        """Random rotation (0, 90, 180, 270 degrees)"""
        angle = np.random.choice([0, 90, 180, 270])
        optical = rotate(optical, angle, reshape=False, mode='reflect')
        radio = rotate(radio, angle, reshape=False, mode='reflect')
        return optical, radio
    
    def _flip(self, optical, radio):
        """Random horizontal/vertical flip"""
        if np.random.rand() < 0.5:
            optical = np.fliplr(optical)
            radio = np.fliplr(radio)
        if np.random.rand() < 0.5:
            optical = np.flipud(optical)
            radio = np.flipud(radio)
        return optical, radio
    
    def _adjust_optical(self, optical):
        """Adjust optical brightness and contrast"""
        # Brightness
        brightness_factor = np.random.uniform(0.8, 1.2)
        optical = optical * brightness_factor
        
        # Contrast
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = optical.mean()
        optical = (optical - mean) * contrast_factor + mean
        
        return optical
    
    def _adjust_radio(self, radio):
        """Adjust radio channel (more conservative due to sparsity)"""
        # Slight brightness adjustment
        brightness_factor = np.random.uniform(0.9, 1.1)
        radio = radio * brightness_factor
        
        # Optional: slight blur to simulate observation conditions
        if np.random.rand() < 0.3:
            sigma = np.random.uniform(0.3, 0.7)
            radio = gaussian_filter(radio, sigma=sigma)
        
        return radio
    
    def _add_noise(self, img, scale=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, scale, img.shape)
        img = img + noise
        return img


class CombinedTransform:
    """Combines preprocessing and augmentation"""
    
    def __init__(self, preprocessing, augmentation=None):
        self.preprocessing = preprocessing
        self.augmentation = augmentation
    
    def __call__(self, optical, radio):
        # Always preprocess
        optical, radio = self.preprocessing(optical, radio)
        
        # Optionally augment
        if self.augmentation is not None:
            optical, radio = self.augmentation(optical, radio)
        
        return optical, radio


if __name__ == "__main__":
    # Test augmentation
    from preprocessing import LoTSSPreprocessing
    from PIL import Image
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    # Load sample
    data_root = Path("data/beautiful_dataset_v2/spiral_galaxies")
    optical_path = list((data_root / "optical").glob("*.png"))[0]
    radio_path = list((data_root / "radio_png").glob("*.png"))[0]
    
    optical = np.array(Image.open(optical_path).convert('L'))
    radio = np.array(Image.open(radio_path).convert('L'))
    
    # Preprocess + augment
    preprocess = LoTSSPreprocessing()
    augment = LoTSSAugmentation(p=1.0)  # Always augment for testing
    transform = CombinedTransform(preprocess, augment)
    
    # Generate multiple augmented versions
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(3):
        opt_aug, rad_aug = transform(optical.copy(), radio.copy())
        
        axes[i, 0].imshow(optical, cmap='gray')
        axes[i, 0].set_title(f'Original Optical {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(opt_aug, cmap='gray')
        axes[i, 1].set_title(f'Augmented Optical {i+1}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(radio, cmap='gray')
        axes[i, 2].set_title(f'Original Radio {i+1}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(rad_aug, cmap='gray')
        axes[i, 3].set_title(f'Augmented Radio {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_test.png', dpi=150)
    print("Saved augmentation_test.png")
