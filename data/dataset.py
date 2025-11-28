"""
LoTSS Dataset Loader
Handles .jpg optical and .png radio images
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter


class LoTSSDataset(Dataset):
    """LoTSS Multi-Modal Dataset (Optical + Radio) - 2 channels"""
    
    def __init__(self, data_root, split='train', transform=None, val_split=0.2, seed=42):
        self.data_root = Path(data_root)
        self.transform = transform
        self.split = split
        
        self.class_folders = [
            'compact_sources',
            'elliptical_galaxies',
            'fr2_radio_galaxies',
            'radio_loud_agn',
            'spiral_galaxies',
            'starburst_galaxies'
        ]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_folders)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        self.samples = self._load_samples()
        
        if len(self.samples) > 0:
            self.samples = self._stratified_split(val_split, seed)
        
        print(f"\n{'='*60}")
        print(f"LoTSS Dataset - {split.upper()} Split")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.samples)}")
        self._print_class_distribution()
        print(f"{'='*60}\n")
    
    def _load_samples(self):
        """Load all optical-radio pairs"""
        samples = []
        
        for class_folder in self.class_folders:
            class_path = self.data_root / class_folder
            optical_dir = class_path / 'optical'
            radio_dir = class_path / 'radio_png'
            
            if not optical_dir.exists() or not radio_dir.exists():
                print(f"  {class_folder:25s}: Missing directories")
                continue
            
            # Get all files (support .jpg, .png, .jpeg)
            optical_files = {}
            for f in optical_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    optical_files[f.stem] = f
            
            radio_files = {}
            for f in radio_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    radio_files[f.stem] = f
            
            # Find matching pairs
            common_stems = set(optical_files.keys()) & set(radio_files.keys())
            
            print(f"  {class_folder:25s}: {len(common_stems)} paired images")
            
            for stem in common_stems:
                samples.append({
                    'optical_path': optical_files[stem],
                    'radio_path': radio_files[stem],
                    'class_name': class_folder,
                    'class_idx': self.class_to_idx[class_folder],
                    'stem': stem
                })
        
        return samples
    
    def _stratified_split(self, val_split, seed):
        """Stratified train/val split"""
        np.random.seed(seed)
        
        class_samples = {cls: [] for cls in self.class_folders}
        for sample in self.samples:
            class_samples[sample['class_name']].append(sample)
        
        split_samples = []
        for cls, cls_samples_list in class_samples.items():
            n = len(cls_samples_list)
            if n == 0:
                continue
            
            n_val = int(n * val_split)
            indices = np.random.permutation(n)
            
            if self.split == 'train':
                selected_indices = indices[n_val:]
            else:
                selected_indices = indices[:n_val]
            
            split_samples.extend([cls_samples_list[i] for i in selected_indices])
        
        return split_samples
    
    def _print_class_distribution(self):
        """Print class distribution"""
        if len(self.samples) == 0:
            print("\nNo samples found!")
            return
        
        class_counts = Counter([s['class_name'] for s in self.samples])
        print("\nClass distribution:")
        for cls in self.class_folders:
            count = class_counts.get(cls, 0)
            percentage = 100 * count / len(self.samples) if self.samples else 0
            print(f"  {cls:25s}: {count:4d} ({percentage:5.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load optical (grayscale)
        optical = Image.open(sample['optical_path']).convert('L')
        optical = np.array(optical, dtype=np.float32)
        
        # Load radio (grayscale)
        radio = Image.open(sample['radio_path']).convert('L')
        radio = np.array(radio, dtype=np.float32)
        
        # Apply transformations
        if self.transform:
            optical, radio = self.transform(optical, radio)
        
        # Stack channels: (2, H, W)
        image = np.stack([optical, radio], axis=0)
        image = torch.from_numpy(image).float()
        
        label = sample['class_idx']
        
        metadata = {
            'class_name': sample['class_name'],
            'stem': sample['stem'],
            'optical_path': str(sample['optical_path']),
            'radio_path': str(sample['radio_path'])
        }
        
        return image, label, metadata
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        if len(self.samples) == 0:
            return torch.ones(len(self.class_folders))
        
        class_counts = Counter([s['class_idx'] for s in self.samples])
        total = len(self.samples)
        
        weights = torch.zeros(len(self.class_folders))
        for idx in range(len(self.class_folders)):
            count = class_counts.get(idx, 1)
            weights[idx] = total / (len(self.class_folders) * count)
        
        return weights


if __name__ == "__main__":
    data_root = Path("data/beautiful_dataset_v2")
    
    print("\nTesting dataset loader...")
    train_ds = LoTSSDataset(data_root, split='train')
    val_ds = LoTSSDataset(data_root, split='val')
    
    if len(train_ds) > 0:
        print(f"\n✓ SUCCESS!")
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")
        
        # Test loading
        img, label, meta = train_ds[0]
        print(f"\n  Sample:")
        print(f"    Shape: {img.shape}")
        print(f"    Label: {label} ({train_ds.idx_to_class[label]})")
        print(f"    File: {meta['stem']}")
