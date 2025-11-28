"""
Base Trainer Class
Common training logic for all models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
import platform
from sklearn.metrics import precision_recall_fscore_support

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics_tracker import MetricsTracker


class BaseTrainer:
    """Base trainer with common functionality"""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Detect Windows
        self.is_windows = platform.system() == 'Windows'
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Compile model for faster training (PyTorch 2.0+, Linux only)
        if (config.get('compile_model', False) 
            and hasattr(torch, 'compile') 
            and not self.is_windows):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        elif self.is_windows and config.get('compile_model', False):
            print("Skipping torch.compile() - not supported on Windows")
        
        # Mixed precision scaler
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Data loaders with optimized settings
        num_workers = config.get('num_workers', 4)
        
        # On Windows, reduce workers if too high
        if self.is_windows and num_workers > 4:
            print(f"Windows detected: reducing num_workers from {num_workers} to 4")
            num_workers = 4
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0 and not self.is_windows,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0 and not self.is_windows,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Loss function (with class weights if provided)
        class_weights = train_dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6)
        )
        
        # Metrics tracker
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = MetricsTracker(
            num_classes=6,
            class_names=train_dataset.class_folders,
            save_dir=self.output_dir
        )
        
        # Save config
        self._save_config()
        
        print(f"\n{'='*60}")
        print("TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Platform: {platform.system()}")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Num Workers: {num_workers}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {config.get('batch_size', 16)}")
        print(f"Learning rate: {config.get('lr', 1e-4)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _save_config(self):
        """Save training configuration"""
        config_copy = self.config.copy()
        config_copy['device'] = str(self.device)
        config_copy['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config_copy['use_amp'] = self.use_amp
        config_copy['platform'] = platform.system()
        
        # Add model info
        params = self.model.get_num_params()
        config_copy['model_params'] = params
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_copy, f, indent=2)
    
    def train_epoch(self, epoch):
        """Train one epoch with mixed precision"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [TRAIN]")
        
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model with mixed precision"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="[VAL]")
            
            for images, labels, _ in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed precision inference
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        # Per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        per_class_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return epoch_loss, epoch_acc, per_class_metrics, all_preds, all_labels
    
    def train(self):
        """Full training loop"""
        num_epochs = self.config.get('num_epochs', 100)
        patience = self.config.get('patience', 15)
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING START - {num_epochs} Epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, per_class_metrics, val_preds, val_labels = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics.update(
                epoch, train_loss, val_loss, train_acc, val_acc,
                current_lr, per_class_metrics
            )
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs} Summary")
            print(f"{'='*60}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            print(f"\nPer-Class Metrics:")
            for i, cls in enumerate(self.train_loader.dataset.class_folders):
                print(f"  {cls:25s}: P={per_class_metrics['precision'][i]:.2%} "
                      f"R={per_class_metrics['recall'][i]:.2%} "
                      f"F1={per_class_metrics['f1'][i]:.2%}")
            
            # Save best model
            if val_acc > self.metrics.best_val_acc - 0.001:
                patience_counter = 0
                self._save_checkpoint('best_model.pth', epoch, val_loss, val_acc)
                print(f"\n>>> NEW BEST MODEL! Val Acc: {val_acc:.2%}")
                
                # Save confusion matrix for best model
                self.metrics.plot_confusion_matrix(val_labels, val_preds)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_loss, val_acc)
            
            print(f"{'='*60}\n")
        
        # Final evaluation
        self._final_evaluation(val_preds, val_labels)
    
    def _save_checkpoint(self, filename, epoch, val_loss, val_acc):
        """Save model checkpoint"""
        # Get base model if compiled
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def _final_evaluation(self, val_preds, val_labels):
        """Final evaluation and visualization"""
        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}\n")
        
        # Generate all plots
        self.metrics.plot_training_curves()
        self.metrics.plot_per_class_metrics()
        self.metrics.plot_confusion_matrix(val_labels, val_preds)
        
        # Save metrics
        self.metrics.save_metrics()
        
        # Classification report
        report = self.metrics.save_classification_report(val_preds, val_labels)
        print(f"\n{report}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best Val Accuracy: {self.metrics.best_val_acc:.2%} (Epoch {self.metrics.best_epoch})")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"{'='*60}\n")