# file: analyse_my_dataset.py
from pathlib import Path

DATASET_ROOT = Path(r"C:\Users\cl502_08\Desktop\LoTSS_data_collection\data\beautiful_dataset_v2")

for class_folder in DATASET_ROOT.iterdir():
    if not class_folder.is_dir():
        continue
    print(f"\n=== {class_folder.name} ===")
    files = list(class_folder.iterdir())
    print(f"Total files: {len(files)}")
    if files:
        print("First 10 filenames:")
        for f in files[:10]:
            print(f"  {f.name}")