"""
Train Vision Transformer on LoTSS Dataset
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.vit_model import ViT_LoTSS
from data.dataset import LoTSSDataset
from data.preprocessing import LoTSSPreprocessing
from data.augmentation import LoTSSAugmentation, CombinedTransform
from training.base_trainer import BaseTrainer


def main():
    parser = argparse.ArgumentParser(description='Train ViT on LoTSS')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/beautiful_dataset_v2',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/vit',
                       help='Output directory')
    
    # Model
    parser.add_argument('--patch_size', type=int, default=30,
                       help='Patch size (600/patch_size should be integer)')
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6,
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=6,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (smaller for ViT)')
    parser.add_argument('--num_epochs', type=int, default=150,
                       help='Number of epochs (ViT needs more)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (lower for ViT)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    
    # Scheduler
    parser.add_argument('--T_0', type=int, default=15,
                       help='Cosine annealing T_0')
    parser.add_argument('--T_mult', type=int, default=2,
                       help='Cosine annealing T_mult')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (longer for ViT)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("VISION TRANSFORMER LOTSS TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
    print("="*60 + "\n")
    
    # Create datasets with stronger augmentation for ViT
    preprocessing = LoTSSPreprocessing(target_size=(600, 600))
    augmentation = LoTSSAugmentation(p=0.9)  # Higher augmentation probability
    
    train_transform = CombinedTransform(preprocessing, augmentation)
    val_transform = CombinedTransform(preprocessing, None)
    
    train_dataset = LoTSSDataset(
        args.data_path,
        split='train',
        transform=train_transform,
        val_split=0.2,
        seed=args.seed
    )
    
    val_dataset = LoTSSDataset(
        args.data_path,
        split='val',
        transform=val_transform,
        val_split=0.2,
        seed=args.seed
    )
    
    # Create model
    model = ViT_LoTSS(
        img_size=600,
        patch_size=args.patch_size,
        in_channels=2,
        num_classes=6,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    params = model.get_num_params()
    print(f"\nModel: Vision Transformer")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Number of patches: {model.patch_embed.n_patches}")
    
    # Create trainer
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'T_0': args.T_0,
        'T_mult': args.T_mult,
        'eta_min': args.eta_min,
        'device': args.device,
        'num_workers': args.num_workers,
        'patience': args.patience,
        'output_dir': args.output_dir,
        'model_name': 'ViT'
    }
    
    trainer = BaseTrainer(model, train_dataset, val_dataset, config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
