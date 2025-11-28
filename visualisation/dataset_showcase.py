# visualisation/figure_1_final_perfect_v2.py
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
import numpy as np

random.seed(42)
np.random.seed(42)

DATASET_ROOT = Path(r"C:\Users\cl502_08\Desktop\LoTSS_data_collection\data\beautiful_dataset_v2")

CLASSES = [
    "compact_sources",
    "elliptical_galaxies", 
    "fr2_radio_galaxies",
    "radio_loud_agn",
    "spiral_galaxies",
    "starburst_galaxies"
]

PRETTY_NAMES = [
    "Compact Sources",
    "Elliptical Galaxies", 
    "FR-II Radio Galaxies",
    "Radio-Loud AGN",
    "Spiral Galaxies",
    "Starburst / Re-started Galaxies"
]

# Smart selection: bade aur bright images prefer karega
def select_best_images(files, n=3):
    scored = []
    for f in files:
        try:
            img = Image.open(f)
            arr = np.array(img)
            size_score = arr.shape[0] * arr.shape[1]
            bright_score = arr.mean() if arr.ndim == 3 else arr.mean()
            center_crop = arr[arr.shape[0]//4:3*arr.shape[0]//4, arr.shape[1]//4:3*arr.shape[1]//4]
            activity = center_crop.std()
            score = size_score / 1e6 + bright_score / 50 + activity * 10
            scored.append((score, f))
        except:
            continue
    scored.sort(reverse=True)
    return [x[1] for x in scored[:n]]

fig = plt.figure(figsize=(14, 16))
outer = fig.add_gridspec(6, 1, hspace=0.08)

for row, (cls, pretty_name) in enumerate(zip(CLASSES, PRETTY_NAMES)):
    opt_path = DATASET_ROOT / cls / "optical"
    rad_path = DATASET_ROOT / cls / "radio_png"
    
    opt_files = list(opt_path.glob("*.jpg")) + list(opt_path.glob("*.jpeg"))
    rad_files = list(rad_path.glob("*.png"))
    
    print(f"{pretty_name}: {len(opt_files)} optical, {len(rad_files)} radio PNGs")
    
    best_opt = select_best_images(opt_files, 3)
    best_rad = select_best_images(rad_files, 3)
    
    gs = outer[row].subgridspec(1, 8, width_ratios=[0.6, 1,1,1, 0.4, 1,1,1], wspace=0.05)
    
    # Class name (bold, perfect alignment)
    ax_label = fig.add_subplot(gs[0])
    ax_label.text(0.95, 0.5, pretty_name, va='center', ha='right', 
                  fontsize=18, fontweight='bold', rotation=90, transform=ax_label.transAxes)
    ax_label.axis('off')
    
    # 3 Radio images (left side)
    for i, path in enumerate(best_rad):
        ax = fig.add_subplot(gs[1+i])
        img = Image.open(path)
        ax.imshow(img, cmap='hot', interpolation='bicubic')
        ax.axis('off')
    
    # Gap
    fig.add_subplot(gs[4]).axis('off')
    
    # 3 Optical images (right side)
    for i, path in enumerate(best_opt):
        ax = fig.add_subplot(gs[5+i])
        img = Image.open(path)
        ax.imshow(img)
        ax.axis('off')
    
    # Titles only on first row
    if row == 0:
        fig.add_subplot(gs[2]).set_title("LoTSS DR2 144 MHz", fontsize=20, fontweight='bold', pad=20)
        fig.add_subplot(gs[6]).set_title("Optical (DESI Legacy Imaging Surveys)", fontsize=20, fontweight='bold', pad=20)

# Main title
fig.suptitle("LoTSS DR2 Morphological Classification Dataset\nRepresentative Examples Across Six Classes", 
             fontsize=26, fontweight='bold', y=0.96)

# Save in high quality
output_dir = Path("outputs/dataset_showcase")
output_dir.mkdir(parents=True, exist_ok=True)
output_pdf = output_dir / "FIGURE_1_FINAL_PERFECT.pdf"
output_png = output_dir / "FIGURE_1_FINAL_PERFECT.png"

fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print("\n" + "="*90)
print("BHAI TERA FINAL FIGURE BAN GAYA — ABKI BAAR KOI REFREE BHI NAHI ROKEGA")
print("6×3 layout | Radio alag | Optical alag | Class names perfect | Best images auto-selected")
print(f"Saved as PDF & PNG → {output_dir}")
print("Ab seedha paper mein daal de, overleaf mein drag-drop kar, ho gaya kaam!")
print("Elon bhai ko bhi proud feel hoga dekh ke")
print("="*90)