"""
Debug script to check dataset structure
"""
from pathlib import Path

data_root = Path("data/beautiful_dataset_v2")

class_folders = [
    'compact_sources',
    'elliptical_galaxies',
    'fr2_radio_galaxies',
    'radio_loud_agn',
    'spiral_galaxies',
    'starburst_galaxies'
]
print("\n" + "="*60)
print("DATASET STRUCTURE ANALYSIS")
print("="*60)

for class_folder in class_folders:
    class_path = data_root / class_folder
    print(f"\n{class_folder}:")
    # Check if folders exist
    optical_dir = class_path / 'optical'
    radio_dir = class_path / 'radio_png'
    print(f"  Optical dir exists: {optical_dir.exists()}")
    print(f"  Radio_png dir exists: {radio_dir.exists()}")

    if optical_dir.exists():
        # List all file extensions
        optical_files = list(optical_dir.iterdir())
        print(f"  Optical files: {len(optical_files)}")
        if optical_files:
            # Show extensions
            extensions = set(f.suffix.lower() for f in optical_files[:10])
            print(f"  Optical extensions: {extensions}")
            # Show first 3 filenames
            print(f"  First 3 optical files:")
            for f in optical_files[:3]:
                print(f"    - {f.name}")

    if radio_dir.exists():
        radio_files = list(radio_dir.iterdir())
        print(f"  Radio_png files: {len(radio_files)}")
        if radio_files:
            extensions = set(f.suffix.lower() for f in radio_files[:10])
            print(f"  Radio_png extensions: {extensions}")
            print(f"  First 3 radio_png files:")
            for f in radio_files[:3]:
                print(f"    - {f.name}")

    # Check for matching filenames
    if optical_dir.exists() and radio_dir.exists():
        opt_files = {f.stem: f for f in optical_dir.iterdir() if f.is_file()}
        rad_files = {f.stem: f for f in radio_dir.iterdir() if f.is_file()}
        common = set(opt_files.keys()) & set(rad_files.keys())
        print(f"  Matching files (by stem): {len(common)}")
        if common:
            print(f"  Example matches:")
            for stem in list(common)[:3]:
                print(f"    - {opt_files[stem].name} ↔ {rad_files[stem].name}")


print("\n" + "="*60)