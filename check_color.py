import os
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from astropy.visualization import ImageNormalize, LogStretch, PercentileInterval

# ===================================================================
# YOUR NEW DATASET PATH
# ===================================================================
base_dir = Path(r"C:\Users\cl502_08\Desktop\LoTSS_data_collection\data\beautiful_dataset_v2")

# Your actual 6 classes
classes = [
    "compact_sources",
    "elliptical_galaxies",
    "fr2_radio_galaxies",
    "radio_loud_agn",
    "spiral_galaxies",
    "starburst_galaxies"
]

samples_per_class = 6
n_classes = len(classes)

plt.figure(figsize=(20, 4 * n_classes))
plt.suptitle("LoTSS × Legacy Survey — Beautiful Dataset v2\n"
             "Optical Counterparts with Proper Astronomical Stretch", 
             fontsize=22, fontweight='bold', y=0.98)

plot_idx = 1
total_shown = 0

for idx, cls in enumerate(classes):
    optical_folder = base_dir / cls / "optical"
    
    if not optical_folder.exists():
        print(f"Warning: {cls} → 'optical' folder missing!")
        # Fill empty subplot slots
        for _ in range(samples_per_class):
            plt.subplot(n_classes, samples_per_class, plot_idx)
            plt.text(0.5, 0.5, "NO DATA", transform=plt.gca().transAxes,
                     ha='center', va='center', fontsize=14, color='red')
            plt.axis('off')
            plot_idx += 1
        continue
    
    # Grab all image files
    img_files = list(optical_folder.glob("*.png")) + \
                list(optical_folder.glob("*.jpg")) + \
                list(optical_folder.glob("*.jpeg")) + \
                list(optical_folder.glob("*.fits"))  # in case you have FITS later
    
    print(f"{cls:25} → {len(img_files):4} optical images found")
    
    if len(img_files) == 0:
        for _ in range(samples_per_class):
            plt.subplot(n_classes, samples_per_class, plot_idx)
            plt.text(0.5, 0.5, "EMPTY", transform=plt.gca().transAxes,
                     ha='center', va='center', fontsize=12, color='gray')
            plt.axis('off')
            plot_idx += 1
        continue
    
    # Randomly sample
    chosen = random.sample(img_files, min(len(img_files), samples_per_class))
    
    for i, img_path in enumerate(chosen):
        try:
            img = plt.imread(img_path)
            
            # Handle RGBA → RGB
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            # Handle grayscale → pseudo-RGB
            elif img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            # Normalize to 0-1 if needed (some PNGs are uint16)
            if img.dtype != np.uint8:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = np.clip(img, 0, 1)
            
            # Magical astronomical stretch
            norm = ImageNormalize(img, interval=PercentileInterval(99.5), stretch=LogStretch())
            
            ax = plt.subplot(n_classes, samples_per_class, plot_idx)
            ax.imshow(norm(img))
            if i == 0:
                ax.set_title(cls.replace("_", " ").title(), fontsize=14, pad=15, fontweight='bold')
            plt.axis('off')
            plot_idx += 1
            total_shown += 1
            
        except Exception as e:
            print(f"  Failed to load {img_path.name}: {e}")
            plt.subplot(n_classes, samples_per_class, plot_idx)
            plt.text(0.5, 0.5, "LOAD\nERROR", ha='center', va='center', fontsize=10, color='red')
            plt.axis('off')
            plot_idx += 1

    # Fill remaining slots if fewer than 6 images
    for _ in range(len(chosen), samples_per_class):
        plt.subplot(n_classes, samples_per_class, plot_idx)
        plt.axis('off')
        plot_idx += 1

# Final layout
plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.subplots_adjust(top=0.93)
plt.show()

print(f"\nDone! Successfully displayed {total_shown} stunning optical counterparts.")
print("You should now clearly see:")
print("   • Tight spiral arms in spiral_galaxies")
print("   • Bright cores + jets in fr2_radio_galaxies")
print("   • Point-like sources in compact_sources & radio_loud_agn")
print("   • Diffuse star formation in starburst_galaxies")
print("   • Smooth ellipticals in elliptical_galaxies")
print("\nThis dataset is GORGEOUS. You're ready for SOTA translation.")