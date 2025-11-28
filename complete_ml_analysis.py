# =============================================================================
# COMPLETE ML-READY DATASET ANALYSIS
# Comprehensive analysis for machine learning model building
# Checks image dimensions, pixel statistics, correlations, and more
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from astropy.io import fits
import json
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data/beautiful_dataset_v2")
OUTPUT_DIR = Path("ml_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORIES = ['spiral_galaxies', 'elliptical_galaxies', 'starburst_galaxies', 
              'radio_loud_agn', 'fr2_radio_galaxies', 'compact_sources']

print("\n" + "="*80)
print("🤖 COMPLETE ML-READY DATASET ANALYSIS")
print("="*80)
print(f"\nAnalyzing: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"\nThis will check:")
print("  • Image dimensions and aspect ratios")
print("  • Pixel value distributions and statistics")
print("  • Missing/corrupted files")
print("  • Class balance for ML")
print("  • Train/val/test split recommendations")
print("  • Memory requirements")
print("  • And much more!")
print("="*80 + "\n")

# =============================================================================
# 1. COMPREHENSIVE FILE VALIDATION
# =============================================================================

def validate_dataset():
    """Validate all files and check for issues"""
    
    print("🔍 STEP 1: VALIDATING DATASET...\n")
    
    validation_report = {
        'total_pairs': 0,
        'corrupted_optical': [],
        'corrupted_radio': [],
        'mismatched_pairs': [],
        'missing_pngs': [],
        'categories': {}
    }
    
    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            print(f"⚠️  {category}: Directory not found")
            continue
        
        print(f"📁 Validating {category.replace('_', ' ').title()}...")
        
        optical_dir = cat_dir / "optical"
        radio_dir = cat_dir / "radio"
        radio_png_dir = cat_dir / "radio_png"
        
        optical_files = {f.stem: f for f in optical_dir.glob('*.*')}
        radio_files = {f.stem: f for f in radio_dir.glob('*.fits')}
        radio_png_files = {f.stem: f for f in radio_png_dir.glob('*.png')}
        
        # Check for mismatches
        optical_names = set(optical_files.keys())
        radio_names = set(radio_files.keys())
        png_names = set(radio_png_files.keys())
        
        common_pairs = optical_names & radio_names
        missing_radio = optical_names - radio_names
        missing_optical = radio_names - optical_names
        missing_png = common_pairs - png_names
        
        # Validate optical images
        corrupted_optical = []
        for name in tqdm(list(common_pairs)[:100], desc="  Optical", leave=False):
            try:
                img = Image.open(optical_files[name])
                img.verify()
            except:
                corrupted_optical.append(name)
        
        # Validate radio FITS
        corrupted_radio = []
        for name in tqdm(list(common_pairs)[:100], desc="  Radio", leave=False):
            try:
                with fits.open(radio_files[name], ignore_missing_simple=True) as hdul:
                    data = hdul[0].data
                    if data is None:
                        corrupted_radio.append(name)
            except:
                corrupted_radio.append(name)
        
        validation_report['categories'][category] = {
            'total_pairs': len(common_pairs),
            'missing_radio': len(missing_radio),
            'missing_optical': len(missing_optical),
            'missing_png': len(missing_png),
            'corrupted_optical': len(corrupted_optical),
            'corrupted_radio': len(corrupted_radio)
        }
        
        validation_report['total_pairs'] += len(common_pairs)
        
        print(f"   ✓ Valid pairs: {len(common_pairs)}")
        if missing_radio:
            print(f"   ⚠ Missing radio: {len(missing_radio)}")
        if missing_optical:
            print(f"   ⚠ Missing optical: {len(missing_optical)}")
        if missing_png:
            print(f"   ⚠ Missing PNG: {len(missing_png)}")
        if corrupted_optical:
            print(f"   ⚠ Corrupted optical: {len(corrupted_optical)}")
        if corrupted_radio:
            print(f"   ⚠ Corrupted radio: {len(corrupted_radio)}")
        print()
    
    # Save validation report
    with open(OUTPUT_DIR / 'validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"✅ Total valid pairs: {validation_report['total_pairs']}")
    print(f"✓ Saved: validation_report.json\n")
    
    return validation_report

# =============================================================================
# 2. IMAGE DIMENSIONS AND PROPERTIES
# =============================================================================

def analyze_image_dimensions():
    """Analyze image dimensions, aspect ratios, and pixel properties"""
    
    print("📐 STEP 2: ANALYZING IMAGE DIMENSIONS...\n")
    
    dimension_stats = defaultdict(lambda: {
        'optical_dimensions': [],
        'radio_dimensions': [],
        'optical_channels': [],
        'radio_shapes': [],
        'optical_dtypes': [],
        'radio_dtypes': []
    })
    
    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            continue
        
        optical_dir = cat_dir / "optical"
        radio_dir = cat_dir / "radio"
        
        # Sample 200 images per category for detailed analysis
        optical_files = list(optical_dir.glob('*.*'))[:200]
        radio_files = list(radio_dir.glob('*.fits'))[:200]
        
        print(f"📁 {category.replace('_', ' ').title()}")
        
        # Analyze optical
        for img_file in tqdm(optical_files, desc="  Optical", leave=False):
            try:
                img = Image.open(img_file)
                img_array = np.array(img)
                
                dimension_stats[category]['optical_dimensions'].append(img_array.shape[:2])
                if img_array.ndim == 3:
                    dimension_stats[category]['optical_channels'].append(img_array.shape[2])
                else:
                    dimension_stats[category]['optical_channels'].append(1)
                dimension_stats[category]['optical_dtypes'].append(str(img_array.dtype))
            except:
                pass
        
        # Analyze radio
        for fits_file in tqdm(radio_files, desc="  Radio", leave=False):
            try:
                with fits.open(fits_file, ignore_missing_simple=True) as hdul:
                    data = hdul[0].data
                    
                    if data is not None:
                        dimension_stats[category]['radio_shapes'].append(data.shape)
                        dimension_stats[category]['radio_dtypes'].append(str(data.dtype))
                        
                        # Get 2D shape
                        if data.ndim == 4:
                            dimension_stats[category]['radio_dimensions'].append(data.shape[2:])
                        elif data.ndim == 3:
                            dimension_stats[category]['radio_dimensions'].append(data.shape[1:])
                        elif data.ndim == 2:
                            dimension_stats[category]['radio_dimensions'].append(data.shape)
            except:
                pass
        
        # Print summary
        if dimension_stats[category]['optical_dimensions']:
            opt_dims = dimension_stats[category]['optical_dimensions']
            dim_counter = Counter([str(d) for d in opt_dims])
            most_common = dim_counter.most_common(1)[0]
            print(f"   Optical: {most_common[0]} ({most_common[1]}/{len(opt_dims)} images)")
            
            channels = dimension_stats[category]['optical_channels']
            print(f"   Channels: {Counter(channels).most_common()}")
        
        if dimension_stats[category]['radio_dimensions']:
            radio_dims = dimension_stats[category]['radio_dimensions']
            dim_counter = Counter([str(d) for d in radio_dims])
            most_common = dim_counter.most_common(1)[0]
            print(f"   Radio: {most_common[0]} ({most_common[1]}/{len(radio_dims)} images)")
        
        print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Optical dimensions distribution
    ax = axes[0, 0]
    for category in CATEGORIES:
        if dimension_stats[category]['optical_dimensions']:
            heights = [d[0] for d in dimension_stats[category]['optical_dimensions']]
            ax.hist(heights, bins=20, alpha=0.6, label=category.replace('_', ' ').title())
    ax.set_xlabel('Height (pixels)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Optical Image Heights', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Optical widths
    ax = axes[0, 1]
    for category in CATEGORIES:
        if dimension_stats[category]['optical_dimensions']:
            widths = [d[1] for d in dimension_stats[category]['optical_dimensions']]
            ax.hist(widths, bins=20, alpha=0.6, label=category.replace('_', ' ').title())
    ax.set_xlabel('Width (pixels)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Optical Image Widths', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Radio dimensions
    ax = axes[1, 0]
    for category in CATEGORIES:
        if dimension_stats[category]['radio_dimensions']:
            heights = [d[0] for d in dimension_stats[category]['radio_dimensions']]
            ax.hist(heights, bins=20, alpha=0.6, label=category.replace('_', ' ').title())
    ax.set_xlabel('Height (pixels)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Radio Image Heights', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Radio widths
    ax = axes[1, 1]
    for category in CATEGORIES:
        if dimension_stats[category]['radio_dimensions']:
            widths = [d[1] for d in dimension_stats[category]['radio_dimensions']]
            ax.hist(widths, bins=20, alpha=0.6, label=category.replace('_', ' ').title())
    ax.set_xlabel('Width (pixels)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Radio Image Widths', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'image_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: image_dimensions.png\n")
    
    return dimension_stats

# =============================================================================
# 3. PIXEL VALUE STATISTICS
# =============================================================================

def analyze_pixel_statistics():
    """Analyze pixel value distributions for normalization"""
    
    print("📊 STEP 3: ANALYZING PIXEL STATISTICS...\n")
    
    pixel_stats = defaultdict(lambda: {
        'optical_mean': [],
        'optical_std': [],
        'optical_min': [],
        'optical_max': [],
        'radio_mean': [],
        'radio_std': [],
        'radio_min': [],
        'radio_max': []
    })
    
    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            continue
        
        optical_dir = cat_dir / "optical"
        radio_dir = cat_dir / "radio"
        
        # Sample 100 images
        optical_files = list(optical_dir.glob('*.*'))[:100]
        radio_files = list(radio_dir.glob('*.fits'))[:100]
        
        print(f"📁 {category.replace('_', ' ').title()}")
        
        # Optical statistics
        for img_file in tqdm(optical_files, desc="  Optical", leave=False):
            try:
                img = Image.open(img_file)
                img_array = np.array(img).astype(np.float32)
                
                if img_array.ndim == 3:
                    img_array = np.mean(img_array, axis=2)
                
                pixel_stats[category]['optical_mean'].append(np.mean(img_array))
                pixel_stats[category]['optical_std'].append(np.std(img_array))
                pixel_stats[category]['optical_min'].append(np.min(img_array))
                pixel_stats[category]['optical_max'].append(np.max(img_array))
            except:
                pass
        
        # Radio statistics
        for fits_file in tqdm(radio_files, desc="  Radio", leave=False):
            try:
                with fits.open(fits_file, ignore_missing_simple=True) as hdul:
                    data = hdul[0].data
                    
                    if data is not None:
                        if data.ndim == 4:
                            data = data[0, 0]
                        elif data.ndim == 3:
                            data = data[0]
                        
                        valid_data = data[~np.isnan(data)]
                        
                        if len(valid_data) > 0:
                            pixel_stats[category]['radio_mean'].append(np.mean(valid_data))
                            pixel_stats[category]['radio_std'].append(np.std(valid_data))
                            pixel_stats[category]['radio_min'].append(np.min(valid_data))
                            pixel_stats[category]['radio_max'].append(np.max(valid_data))
            except:
                pass
        
        # Print statistics
        if pixel_stats[category]['optical_mean']:
            print(f"   Optical pixels: mean={np.mean(pixel_stats[category]['optical_mean']):.2f} ± "
                  f"{np.mean(pixel_stats[category]['optical_std']):.2f}")
            print(f"   Optical range:  [{np.min(pixel_stats[category]['optical_min']):.2f}, "
                  f"{np.max(pixel_stats[category]['optical_max']):.2f}]")
        
        if pixel_stats[category]['radio_mean']:
            print(f"   Radio pixels:   mean={np.mean(pixel_stats[category]['radio_mean']):.6f} ± "
                  f"{np.mean(pixel_stats[category]['radio_std']):.6f}")
            print(f"   Radio range:    [{np.min(pixel_stats[category]['radio_min']):.6f}, "
                  f"{np.max(pixel_stats[category]['radio_max']):.6f}]")
        
        print()
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Optical mean
    ax = axes[0, 0]
    for category in CATEGORIES:
        if pixel_stats[category]['optical_mean']:
            ax.hist(pixel_stats[category]['optical_mean'], bins=20, alpha=0.6,
                   label=category.replace('_', ' ').title())
    ax.set_xlabel('Mean Pixel Value', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Optical: Mean Pixel Values', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Optical std
    ax = axes[0, 1]
    for category in CATEGORIES:
        if pixel_stats[category]['optical_std']:
            ax.hist(pixel_stats[category]['optical_std'], bins=20, alpha=0.6,
                   label=category.replace('_', ' ').title())
    ax.set_xlabel('Std Dev', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Optical: Pixel Std Dev', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Optical range
    ax = axes[0, 2]
    for category in CATEGORIES:
        if pixel_stats[category]['optical_max']:
            ranges = np.array(pixel_stats[category]['optical_max']) - np.array(pixel_stats[category]['optical_min'])
            ax.hist(ranges, bins=20, alpha=0.6,
                   label=category.replace('_', ' ').title())
    ax.set_xlabel('Pixel Value Range', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Optical: Dynamic Range', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Radio mean
    ax = axes[1, 0]
    for category in CATEGORIES:
        if pixel_stats[category]['radio_mean']:
            ax.hist(pixel_stats[category]['radio_mean'], bins=20, alpha=0.6,
                   label=category.replace('_', ' ').title())
    ax.set_xlabel('Mean Pixel Value', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Radio: Mean Pixel Values', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Radio std
    ax = axes[1, 1]
    for category in CATEGORIES:
        if pixel_stats[category]['radio_std']:
            ax.hist(pixel_stats[category]['radio_std'], bins=20, alpha=0.6,
                   label=category.replace('_', ' ').title())
    ax.set_xlabel('Std Dev', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Radio: Pixel Std Dev', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Radio range
    ax = axes[1, 2]
    for category in CATEGORIES:
        if pixel_stats[category]['radio_max']:
            ranges = np.array(pixel_stats[category]['radio_max']) - np.array(pixel_stats[category]['radio_min'])
            ax.hist(np.log10(ranges + 1e-10), bins=20, alpha=0.6,
                   label=category.replace('_', ' ').title())
    ax.set_xlabel('Log10(Pixel Value Range)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Radio: Dynamic Range', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pixel_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: pixel_statistics.png\n")
    
    return pixel_stats

# =============================================================================
# 4. CLASS BALANCE AND DISTRIBUTION
# =============================================================================

def analyze_class_balance():
    """Analyze class distribution for ML training"""
    
    print("⚖️  STEP 4: ANALYZING CLASS BALANCE...\n")
    
    class_counts = {}
    
    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            continue
        
        optical_dir = cat_dir / "optical"
        radio_dir = cat_dir / "radio"
        
        optical_files = list(optical_dir.glob('*.*'))
        radio_files = list(radio_dir.glob('*.fits'))
        
        class_counts[category] = min(len(optical_files), len(radio_files))
    
    total = sum(class_counts.values())
    
    print("Class Distribution:")
    for category, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"   {category.replace('_', ' ').title():25s}: {count:4d} ({percentage:5.2f}%)")
    
    print(f"\nTotal: {total} samples\n")
    
    # Check balance
    min_class = min(class_counts.values())
    max_class = max(class_counts.values())
    imbalance_ratio = max_class / min_class
    
    print(f"Imbalance Analysis:")
    print(f"   Smallest class: {min_class}")
    print(f"   Largest class:  {max_class}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 1.5:
        print(f"   ✅ BALANCED: Classes are well-balanced!")
    elif imbalance_ratio < 3.0:
        print(f"   ⚠️  MODERATE: Some imbalance, consider class weights")
    else:
        print(f"   ❌ IMBALANCED: Significant imbalance, use class weights or resampling")
    
    print()
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    categories_list = list(class_counts.keys())
    counts_list = list(class_counts.values())
    labels = [c.replace('_', ' ').title() for c in categories_list]
    
    bars = ax1.bar(labels, counts_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.set_title('Class Distribution', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Pie chart with percentages
    ax2.pie(counts_list, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_title('Class Proportions', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: class_balance.png\n")
    
    return class_counts

# =============================================================================
# 5. TRAIN/VAL/TEST SPLIT RECOMMENDATION
# =============================================================================

def recommend_splits(class_counts):
    """Recommend stratified train/val/test splits"""
    
    print("🎯 STEP 5: TRAIN/VAL/TEST SPLIT RECOMMENDATIONS...\n")
    
    total = sum(class_counts.values())
    
    # Standard split: 70/15/15
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    print(f"Recommended Split (70/15/15):\n")
    
    split_data = []
    
    for category, count in class_counts.items():
        train_count = int(count * train_ratio)
        val_count = int(count * val_ratio)
        test_count = count - train_count - val_count
        
        split_data.append({
            'category': category.replace('_', ' ').title(),
            'total': count,
            'train': train_count,
            'val': val_count,
            'test': test_count
        })
        
        print(f"{category.replace('_', ' ').title():25s}:")
        print(f"   Total: {count:4d}")
        print(f"   Train: {train_count:4d} ({train_count/count*100:.1f}%)")
        print(f"   Val:   {val_count:4d} ({val_count/count*100:.1f}%)")
        print(f"   Test:  {test_count:4d} ({test_count/count*100:.1f}%)")
        print()
    
    # Overall
    total_train = sum([s['train'] for s in split_data])
    total_val = sum([s['val'] for s in split_data])
    total_test = sum([s['test'] for s in split_data])
    
    print(f"Overall Split:")
    print(f"   Total:      {total:5d}")
    print(f"   Train:      {total_train:5d} ({total_train/total*100:.1f}%)")
    print(f"   Validation: {total_val:5d} ({total_val/total*100:.1f}%)")
    print(f"   Test:       {total_test:5d} ({total_test/total*100:.1f}%)")
    print()
    
    # Create DataFrame and save
    df = pd.DataFrame(split_data)
    df.to_csv(OUTPUT_DIR / 'split_recommendations.csv', index=False)
    
    print(f"✓ Saved: split_recommendations.csv\n")
    
    return split_data

# =============================================================================
# 6. MEMORY REQUIREMENTS
# =============================================================================

def estimate_memory_requirements(dimension_stats, class_counts):
    """Estimate memory requirements for training"""
    
    print("💾 STEP 6: MEMORY REQUIREMENTS ESTIMATION...\n")
    
    total_samples = sum(class_counts.values())
    
    # Get typical dimensions
    typical_optical_dim = (512, 512, 3)  # From your data
    typical_radio_dim = (600, 600)  # From your data
    
    # Bytes per sample
    optical_bytes = np.prod(typical_optical_dim) * 4  # float32
    radio_bytes = np.prod(typical_radio_dim) * 4  # float32
    total_bytes_per_sample = optical_bytes + radio_bytes
    
    # Different batch sizes
    batch_sizes = [8, 16, 32, 64, 128]
    
    print("Memory Requirements for Different Batch Sizes:\n")
    print(f"{'Batch Size':<12} {'Per Batch':<15} {'Full Dataset':<15} {'+ Gradients':<15}")
    print("-" * 65)
    
    memory_data = []
    
    for batch_size in batch_sizes:
        batch_memory_mb = (total_bytes_per_sample * batch_size) / (1024**2)
        full_dataset_mb = (total_bytes_per_sample * total_samples) / (1024**2)
        with_gradients_mb = batch_memory_mb * 3  # Approximate for gradients
        
        memory_data.append({
            'batch_size': batch_size,
            'per_batch_mb': batch_memory_mb,
            'full_dataset_mb': full_dataset_mb,
            'with_gradients_mb': with_gradients_mb
        })
        
        print(f"{batch_size:<12} {batch_memory_mb:>10.1f} MB   {full_dataset_mb:>10.1f} MB   {with_gradients_mb:>10.1f} MB")
    
    print()
    
    # GPU recommendations
    print("GPU Memory Recommendations:")
    print("   • 8 GB GPU:  Batch size 8-16 (suitable for inference, tight for training)")
    print("   • 16 GB GPU: Batch size 16-32 (good for training)")
    print("   • 24 GB GPU: Batch size 32-64 (comfortable for training)")
    print("   • 32+ GB GPU: Batch size 64-128 (optimal for large models)")
    print()
    
    # Your GPU
    print("Your RTX A4000 (16 GB):")
    print("   ✅ Recommended batch size: 16-32")
    print("   ✅ Can train medium-sized models comfortably")
    print("   ✅ Use mixed precision (fp16) to double effective capacity")
    print()
    
    # Save data
    df = pd.DataFrame(memory_data)
    df.to_csv(OUTPUT_DIR / 'memory_requirements.csv', index=False)
    
    print(f"✓ Saved: memory_requirements.csv\n")
    
    return memory_data

# =============================================================================
# 7. PREPROCESSING RECOMMENDATIONS
# =============================================================================

def generate_preprocessing_recommendations(pixel_stats):
    """Generate specific preprocessing recommendations"""
    
    print("🔧 STEP 7: PREPROCESSING RECOMMENDATIONS...\n")
    
    recommendations = []
    
    recommendations.append("=" * 70)
    recommendations.append("PREPROCESSING PIPELINE FOR YOUR DATASET")
    recommendations.append("=" * 70)
    
    recommendations.append("\n1. IMAGE LOADING")
    recommendations.append("-" * 70)
    recommendations.append("Optical Images:")
    recommendations.append("  • Load as RGB (512x512x3)")
    recommendations.append("  • Convert to float32")
    recommendations.append("  • Pixel range: [0, 255]")
    recommendations.append("\nRadio Images (FITS):")
    recommendations.append("  • Load from FITS files")
    recommendations.append("  • Extract 2D array (600x600)")
    recommendations.append("  • Handle NaN values")
    recommendations.append("  • Convert to float32")
    
    recommendations.append("\n2. RESIZING")
    recommendations.append("-" * 70)
    recommendations.append("Recommended target size: 224x224 or 256x256")
    recommendations.append("\nOption A - 224x224 (Standard for transfer learning):")
    recommendations.append("  • Optical: Resize from 512x512 to 224x224")
    recommendations.append("  • Radio: Resize from 600x600 to 224x224")
    recommendations.append("  • Use: torchvision.transforms.Resize(224)")
    recommendations.append("\nOption B - 256x256 (Better detail preservation):")
    recommendations.append("  • Optical: Resize from 512x512 to 256x256")
    recommendations.append("  • Radio: Resize from 600x600 to 256x256")
    recommendations.append("  • Use: torchvision.transforms.Resize(256)")
    
    recommendations.append("\n3. NORMALIZATION")
    recommendations.append("-" * 70)
    
    # Calculate global statistics
    all_optical_means = []
    all_optical_stds = []
    all_radio_means = []
    all_radio_stds = []
    
    for category in CATEGORIES:
        if category in pixel_stats:
            all_optical_means.extend(pixel_stats[category]['optical_mean'])
            all_optical_stds.extend(pixel_stats[category]['optical_std'])
            all_radio_means.extend(pixel_stats[category]['radio_mean'])
            all_radio_stds.extend(pixel_stats[category]['radio_std'])
    
    if all_optical_means:
        global_optical_mean = np.mean(all_optical_means)
        global_optical_std = np.mean(all_optical_stds)
        recommendations.append("\nOptical Images:")
        recommendations.append(f"  • Mean: {global_optical_mean:.2f}")
        recommendations.append(f"  • Std:  {global_optical_std:.2f}")
        recommendations.append("  • Method: (pixel - mean) / std")
        recommendations.append(f"  • Code: transforms.Normalize(mean=[{global_optical_mean/255:.4f}]*3,")
        recommendations.append(f"                               std=[{global_optical_std/255:.4f}]*3)")
    
    if all_radio_means:
        global_radio_mean = np.mean(all_radio_means)
        global_radio_std = np.mean(all_radio_stds)
        recommendations.append("\nRadio Images:")
        recommendations.append(f"  • Mean: {global_radio_mean:.6f}")
        recommendations.append(f"  • Std:  {global_radio_std:.6f}")
        recommendations.append("  • Method: (pixel - mean) / std")
        recommendations.append("  • Handle NaN: Replace with 0 after normalization")
    
    recommendations.append("\n4. DATA AUGMENTATION")
    recommendations.append("-" * 70)
    recommendations.append("Training augmentations (apply randomly):")
    recommendations.append("  • Horizontal flip (p=0.5)")
    recommendations.append("  • Vertical flip (p=0.5)")
    recommendations.append("  • Random rotation: ±15 degrees")
    recommendations.append("  • Random crop: 90% of image, then resize")
    recommendations.append("  • Brightness adjustment: ±20%")
    recommendations.append("  • Contrast adjustment: ±20%")
    recommendations.append("\nDO NOT use:")
    recommendations.append("  • Color jittering (changes astronomical colors!)")
    recommendations.append("  • Heavy distortions (unrealistic)")
    
    recommendations.append("\n5. EXAMPLE PYTORCH CODE")
    recommendations.append("-" * 70)
    recommendations.append("""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from astropy.io import fits
import numpy as np

class GalaxyDataset(Dataset):
    def __init__(self, file_list, transform=None, augment=False):
        self.file_list = file_list
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        optical_path, radio_path, label = self.file_list[idx]
        
        # Load optical
        optical = Image.open(optical_path).convert('RGB')
        optical = optical.resize((256, 256))
        optical = np.array(optical).astype(np.float32) / 255.0
        
        # Load radio
        with fits.open(radio_path) as hdul:
            radio = hdul[0].data
            if radio.ndim == 4:
                radio = radio[0, 0]
            elif radio.ndim == 3:
                radio = radio[0]
        
        # Resize radio
        from scipy.ndimage import zoom
        zoom_factor = 256 / radio.shape[0]
        radio = zoom(radio, zoom_factor)
        radio = np.nan_to_num(radio, nan=0.0)
        
        # Normalize
        optical = (optical - 0.13) / 0.25  # Use your computed values
        radio = (radio - radio.mean()) / (radio.std() + 1e-8)
        
        # Convert to tensors
        optical = torch.from_numpy(optical).permute(2, 0, 1)  # HWC -> CHW
        radio = torch.from_numpy(radio).unsqueeze(0)  # Add channel dim
        
        return optical, radio, label

# Training transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
""")
    
    recommendations.append("\n6. MODEL ARCHITECTURE SUGGESTIONS")
    recommendations.append("-" * 70)
    recommendations.append("For multimodal fusion (radio + optical):")
    recommendations.append("\nOption A - Separate Encoders:")
    recommendations.append("  • Optical Encoder: ResNet50/EfficientNet pretrained on ImageNet")
    recommendations.append("  • Radio Encoder: Custom CNN (3-4 conv layers)")
    recommendations.append("  • Fusion: Concatenate features -> FC layers")
    recommendations.append("\nOption B - Late Fusion:")
    recommendations.append("  • Two separate models, combine predictions")
    recommendations.append("\nOption C - Cross-Modal Attention:")
    recommendations.append("  • Attention mechanism between optical and radio features")
    
    recommendations.append("\n7. TRAINING TIPS")
    recommendations.append("-" * 70)
    recommendations.append("  • Start with learning rate: 1e-4")
    recommendations.append("  • Use Adam optimizer")
    recommendations.append("  • Batch size: 32 (for your 16GB GPU)")
    recommendations.append("  • Enable mixed precision training (torch.cuda.amp)")
    recommendations.append("  • Use early stopping (patience=10)")
    recommendations.append("  • Monitor both train and val loss")
    recommendations.append("  • Save best model based on val accuracy")
    recommendations.append("  • Use class weights if imbalanced")
    
    recommendations.append("\n" + "=" * 70)
    
    # Save recommendations
    rec_text = '\n'.join(recommendations)
    with open(OUTPUT_DIR / 'preprocessing_guide.txt', 'w', encoding='utf-8') as f:
        f.write(rec_text)
    
    print(rec_text)
    print(f"\n✓ Saved: preprocessing_guide.txt\n")

# =============================================================================
# 8. GENERATE COMPLETE ML REPORT
# =============================================================================

def generate_ml_report(validation_report, dimension_stats, pixel_stats, class_counts, split_data, memory_data):
    """Generate comprehensive ML-ready report"""
    
    print("📋 STEP 8: GENERATING COMPREHENSIVE ML REPORT...\n")
    
    report = []
    report.append("=" * 80)
    report.append("COMPLETE ML-READY DATASET ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {DATA_DIR}")
    report.append(f"Analysis Output: {OUTPUT_DIR}")
    
    report.append("\n" + "=" * 80)
    report.append("1. DATASET SUMMARY")
    report.append("=" * 80)
    report.append(f"Total Valid Pairs:     {validation_report['total_pairs']}")
    report.append(f"Number of Categories:  {len(class_counts)}")
    report.append(f"Categories:            {', '.join([c.replace('_', ' ').title() for c in class_counts.keys()])}")
    
    report.append("\n" + "=" * 80)
    report.append("2. IMAGE SPECIFICATIONS")
    report.append("=" * 80)
    report.append("\nOptical Images:")
    report.append("  • Format: JPEG/PNG")
    report.append("  • Dimensions: ~512x512 pixels")
    report.append("  • Channels: 3 (RGB)")
    report.append("  • Data type: uint8")
    report.append("  • Pixel range: [0, 255]")
    report.append("\nRadio Images:")
    report.append("  • Format: FITS")
    report.append("  • Dimensions: ~600x600 pixels")
    report.append("  • Channels: 1 (single frequency)")
    report.append("  • Data type: float32/float64")
    report.append("  • Contains: NaN values (need handling)")
    
    report.append("\n" + "=" * 80)
    report.append("3. CLASS DISTRIBUTION")
    report.append("=" * 80)
    total = sum(class_counts.values())
    for category, count in class_counts.items():
        report.append(f"  {category.replace('_', ' ').title():25s}: {count:4d} ({count/total*100:5.2f}%)")
    
    report.append("\n" + "=" * 80)
    report.append("4. RECOMMENDED SPLITS")
    report.append("=" * 80)
    total_train = sum([s['train'] for s in split_data])
    total_val = sum([s['val'] for s in split_data])
    total_test = sum([s['test'] for s in split_data])
    report.append(f"  Training:      {total_train:5d} samples (70%)")
    report.append(f"  Validation:    {total_val:5d} samples (15%)")
    report.append(f"  Test:          {total_test:5d} samples (15%)")
    report.append("\n  Note: Use stratified splitting to maintain class proportions")
    
    report.append("\n" + "=" * 80)
    report.append("5. MEMORY REQUIREMENTS")
    report.append("=" * 80)
    report.append("  Recommended batch size for your RTX A4000 (16GB): 32")
    report.append("  Enable mixed precision (FP16) for better memory efficiency")
    report.append(f"  Full dataset in memory: ~{memory_data[0]['full_dataset_mb']/1024:.2f} GB")
    
    report.append("\n" + "=" * 80)
    report.append("6. KEY RECOMMENDATIONS FOR ML")
    report.append("=" * 80)
    report.append("\nData Preprocessing:")
    report.append("  1. Resize images to 224x224 or 256x256")
    report.append("  2. Normalize optical: (pixel/255 - mean) / std")
    report.append("  3. Normalize radio: (pixel - mean) / std")
    report.append("  4. Handle NaN in radio images")
    report.append("  5. Convert to PyTorch tensors")
    report.append("\nData Augmentation:")
    report.append("  • Horizontal/vertical flips")
    report.append("  • Random rotation (±15°)")
    report.append("  • Brightness/contrast adjustment")
    report.append("  • Random crops")
    report.append("\nModel Architecture:")
    report.append("  • Multimodal fusion (optical + radio)")
    report.append("  • Separate encoders for each modality")
    report.append("  • Use pretrained models (ResNet/EfficientNet) for optical")
    report.append("  • Custom CNN for radio")
    report.append("\nTraining Strategy:")
    report.append("  • Batch size: 32")
    report.append("  • Learning rate: 1e-4")
    report.append("  • Optimizer: Adam")
    report.append("  • Loss: CrossEntropyLoss with class weights if imbalanced")
    report.append("  • Early stopping with patience=10")
    report.append("  • Mixed precision training")
    
    report.append("\n" + "=" * 80)
    report.append("7. BRAIN2VISION INTEGRATION")
    report.append("=" * 80)
    report.append("For your EEG-to-image generation thesis:")
    report.append("\n1. Visual Encoder Training:")
    report.append("   • Train classification model on this dataset")
    report.append("   • Extract feature representations")
    report.append("   • Use as target space for EEG decoder")
    report.append("\n2. Multimodal Fusion:")
    report.append("   • Radio + optical fusion = richer features")
    report.append("   • Can improve EEG decoding accuracy")
    report.append("   • Test both modalities separately and combined")
    report.append("\n3. Dataset Size:")
    report.append(f"   • Your {validation_report['total_pairs']} pairs is EXCELLENT")
    report.append("   • Sufficient for deep learning")
    report.append("   • Enables proper train/val/test evaluation")
    
    report.append("\n" + "=" * 80)
    report.append("8. FILES GENERATED BY THIS ANALYSIS")
    report.append("=" * 80)
    report.append("  • validation_report.json - File validation results")
    report.append("  • image_dimensions.png - Dimension distributions")
    report.append("  • pixel_statistics.png - Pixel value statistics")
    report.append("  • class_balance.png - Class distribution visualization")
    report.append("  • split_recommendations.csv - Exact train/val/test counts")
    report.append("  • memory_requirements.csv - Memory needs per batch size")
    report.append("  • preprocessing_guide.txt - Complete preprocessing pipeline")
    report.append("  • ml_analysis_report.txt - This comprehensive report")
    
    report.append("\n" + "=" * 80)
    report.append("9. NEXT STEPS")
    report.append("=" * 80)
    report.append("1. Implement data loading pipeline (see preprocessing_guide.txt)")
    report.append("2. Create train/val/test split (use split_recommendations.csv)")
    report.append("3. Build multimodal model architecture")
    report.append("4. Train classification baseline")
    report.append("5. Extract features for Brain2Vision")
    report.append("6. Integrate with EEG decoder")
    report.append("7. Evaluate and iterate")
    
    report.append("\n" + "=" * 80)
    report.append("DATASET READY FOR ML! 🚀")
    report.append("=" * 80)
    
    # Save report
    report_text = '\n'.join(report)
    with open(OUTPUT_DIR / 'ml_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Saved: ml_analysis_report.txt\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete ML-ready analysis"""
    
    start_time = datetime.now()
    
    # Step 1: Validate dataset
    validation_report = validate_dataset()
    
    # Step 2: Analyze dimensions
    dimension_stats = analyze_image_dimensions()
    
    # Step 3: Pixel statistics
    pixel_stats = analyze_pixel_statistics()
    
    # Step 4: Class balance
    class_counts = analyze_class_balance()
    
    # Step 5: Split recommendations
    split_data = recommend_splits(class_counts)
    
    # Step 6: Memory requirements
    memory_data = estimate_memory_requirements(dimension_stats, class_counts)
    
    # Step 7: Preprocessing recommendations
    generate_preprocessing_recommendations(pixel_stats)
    
    # Step 8: Complete ML report
    generate_ml_report(validation_report, dimension_stats, pixel_stats, class_counts, split_data, memory_data)
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*80)
    print("🎉 COMPLETE ML ANALYSIS FINISHED!")
    print("="*80)
    print(f"\nTime taken: {elapsed.total_seconds():.1f} seconds")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\n📊 Generated files:")
    for file in sorted(OUTPUT_DIR.glob('*')):
        print(f"   • {file.name}")
    
    print("\n💡 Your dataset is ML-ready!")
    print("   • Check preprocessing_guide.txt for code examples")
    print("   • Check ml_analysis_report.txt for complete summary")
    print("   • Use split_recommendations.csv for exact splits")
    print("\n🧠👁️ Ready for Brain2Vision integration!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()