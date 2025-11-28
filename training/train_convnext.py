"""
Train ConvNeXt-Tiny on LoTSS Dataset
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.convnext_model import ConvNeXtTiny_LoTSS
from data.dataset import LoTSSDataset
from data.preprocessing import LoTSSPreprocessing
from data.augmentation import LoTSSAugmentation, CombinedTransform
from training.base_trainer import BaseTrainer


def main():
    parser = argparse.ArgumentParser(description='Train ConvNeXt-Tiny on LoTSS')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/beautiful_dataset_v2',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/convnext',
                       help='Output directory')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=12,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Scheduler
    parser.add_argument('--T_0', type=int, default=10,
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
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("CONVNEXT-TINY LOTSS TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    for arg in vars(args):
        print(f"  {arg:20s}: {getattr(args, arg)}")
    print("="*60 + "\n")
    
    # Create datasets
    preprocessing = LoTSSPreprocessing(target_size=(600, 600))
    augmentation = LoTSSAugmentation(p=0.7)
    
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
    model = ConvNeXtTiny_LoTSS(
        num_classes=6,
        pretrained=True,
        dropout=args.dropout
    )
    
    params = model.get_num_params()
    print(f"\nModel: ConvNeXt-Tiny")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
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
        'model_name': 'ConvNeXt-Tiny'
    }
    
    trainer = BaseTrainer(model, train_dataset, val_dataset, config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()