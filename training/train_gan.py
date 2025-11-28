"""
Train Pix2Pix GAN for Radio <-> Optical Translation
Optimized for 16GB GPU with 600x600 images
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import platform
import gc
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.spadegan_model import UNetGenerator, PatchGANDiscriminator, VGGLoss
from data.dataset import LoTSSDataset
from data.preprocessing import LoTSSPreprocessing
from data.augmentation import CombinedTransform


# ============================================================================
# METRICS TRACKER
# ============================================================================

class GANMetricsTracker:
    """Track GAN training metrics"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.g_losses = []
        self.d_losses = []
        self.g_adv_losses = []
        self.g_l1_losses = []
        self.g_vgg_losses = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.g_lrs = []
        self.d_lrs = []
        self.best_psnr = 0.0
        self.best_epoch = 0
    
    def update(self, g_loss_dict, d_loss, g_lr, d_lr):
        self.g_losses.append(g_loss_dict['total'])
        self.d_losses.append(d_loss)
        self.g_adv_losses.append(g_loss_dict.get('adv', 0))
        self.g_l1_losses.append(g_loss_dict.get('l1', 0))
        self.g_vgg_losses.append(g_loss_dict.get('vgg', 0))
        self.g_lrs.append(g_lr)
        self.d_lrs.append(d_lr)
    
    def add_image_metrics(self, psnr_val, ssim_val, epoch):
        self.psnr_scores.append(psnr_val)
        self.ssim_scores.append(ssim_val)
        if psnr_val > self.best_psnr:
            self.best_psnr = psnr_val
            self.best_epoch = epoch
            return True
        return False
    
    def plot_curves(self):
        """Plot comprehensive GAN training curves"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(self.g_losses) + 1)
        
        # G vs D Loss
        axes[0, 0].plot(epochs, self.g_losses, 'b-', label='Generator', linewidth=2)
        axes[0, 0].plot(epochs, self.d_losses, 'r-', label='Discriminator', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator vs Discriminator Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # G Loss Components
        axes[0, 1].plot(epochs, self.g_adv_losses, label='Adversarial', linewidth=2)
        axes[0, 1].plot(epochs, self.g_l1_losses, label='L1', linewidth=2)
        axes[0, 1].plot(epochs, self.g_vgg_losses, label='VGG', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Generator Loss Components', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rates
        axes[0, 2].plot(epochs, self.g_lrs, label='G LR', linewidth=2)
        axes[0, 2].plot(epochs, self.d_lrs, label='D LR', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[0, 2].set_yscale('log')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # PSNR
        if self.psnr_scores:
            val_epochs = range(1, len(self.psnr_scores) + 1)
            axes[1, 0].plot(val_epochs, self.psnr_scores, 'g-', linewidth=2, marker='o')
            axes[1, 0].axhline(y=self.best_psnr, color='r', linestyle='--', 
                              label=f'Best: {self.best_psnr:.2f}dB')
            axes[1, 0].set_xlabel('Validation Point')
            axes[1, 0].set_ylabel('PSNR (dB)')
            axes[1, 0].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # SSIM
            axes[1, 1].plot(val_epochs, self.ssim_scores, 'm-', linewidth=2, marker='o')
            best_ssim = max(self.ssim_scores) if self.ssim_scores else 0
            axes[1, 1].axhline(y=best_ssim, color='r', linestyle='--', 
                              label=f'Best: {best_ssim:.4f}')
            axes[1, 1].set_xlabel('Validation Point')
            axes[1, 1].set_ylabel('SSIM')
            axes[1, 1].set_title('Structural Similarity Index', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Loss Ratio
        ratio = [g / (d + 1e-8) for g, d in zip(self.g_losses, self.d_losses)]
        axes[1, 2].plot(epochs, ratio, 'purple', linewidth=2)
        axes[1, 2].axhline(y=1, color='k', linestyle='--', label='Balanced')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('G Loss / D Loss')
        axes[1, 2].set_title('Loss Ratio (G/D)', fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()
        print(f"  Saved: {self.save_dir / 'training_curves.png'}")
    
    def save_metrics(self):
        metrics = {
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'g_adv_losses': self.g_adv_losses,
            'g_l1_losses': self.g_l1_losses,
            'g_vgg_losses': self.g_vgg_losses,
            'psnr_scores': self.psnr_scores,
            'ssim_scores': self.ssim_scores,
            'best_psnr': self.best_psnr,
            'best_epoch': self.best_epoch
        }
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_metrics(fake, real):
    """Compute PSNR and SSIM"""
    fake_np = (fake.detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    real_np = (real.detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    
    psnr_vals = []
    ssim_vals = []
    for f, r in zip(fake_np, real_np):
        f_img = f.squeeze()
        r_img = r.squeeze()
        psnr_vals.append(psnr(r_img, f_img, data_range=1.0))
        ssim_vals.append(ssim(r_img, f_img, data_range=1.0))
    
    return np.mean(psnr_vals), np.mean(ssim_vals)


def save_samples(input_img, generated, real, epoch, save_dir, mode='radio2optical'):
    """Save sample images at 600x600"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(4, input_img.size(0))
    fig, axes = plt.subplots(4, n_samples, figsize=(4 * n_samples, 16))
    
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    if mode == 'radio2optical':
        titles = ['Radio Input', 'Generated Optical', 'Real Optical', 'Difference']
    else:
        titles = ['Optical Input', 'Generated Radio', 'Real Radio', 'Difference']
    
    for i in range(n_samples):
        # Input
        img = input_img[i].detach().cpu().squeeze().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(titles[0] if i == 0 else '', fontsize=12)
        axes[0, i].axis('off')
        
        # Generated
        img = generated[i].detach().cpu().squeeze().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(titles[1] if i == 0 else '', fontsize=12)
        axes[1, i].axis('off')
        
        # Real
        img = real[i].detach().cpu().squeeze().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(titles[2] if i == 0 else '', fontsize=12)
        axes[2, i].axis('off')
        
        # Difference
        gen = generated[i].detach().cpu().squeeze().numpy()
        rl = real[i].detach().cpu().squeeze().numpy()
        diff = np.abs(gen - rl)
        axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[3, i].set_title(titles[3] if i == 0 else '', fontsize=12)
        axes[3, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - 600x600 Resolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'samples_epoch_{epoch:04d}.png', dpi=100)
    plt.close()


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix GAN on LoTSS (600x600)')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/beautiful_dataset_v2')
    parser.add_argument('--output_dir', type=str, default='outputs/gan')
    parser.add_argument('--mode', type=str, default='radio2optical',
                       choices=['radio2optical', 'optical2radio'])
    
    # Model
    parser.add_argument('--ngf', type=int, default=64, help='Generator filters')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator filters')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--grad_accum', type=int, default=2)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    is_windows = platform.system() == 'Windows'
    
    output_dir = Path(args.output_dir)
    (output_dir / 'samples').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    
    num_workers = min(args.num_workers, 4) if is_windows else args.num_workers
    
    # Save config
    config = vars(args).copy()
    config['platform'] = platform.system()
    config['image_size'] = 600
    config['architecture'] = 'Pix2Pix U-Net'
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"PIX2PIX GAN TRAINING - {args.mode.upper()}")
    print("=" * 60)
    print(f"  Architecture: U-Net Generator + PatchGAN Discriminator")
    print(f"  Image Size: 600x600 (preserved)")
    print(f"  Platform: {platform.system()}")
    print(f"  Mixed Precision: {args.use_amp}")
    print(f"  Effective Batch Size: {args.batch_size * args.grad_accum}")
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print("=" * 60 + "\n")
    
    # Dataset
    preprocessing = LoTSSPreprocessing(target_size=(600, 600))
    transform = CombinedTransform(preprocessing, None)
    
    train_dataset = LoTSSDataset(
        args.data_path, split='train', transform=transform,
        val_split=0.1, seed=args.seed
    )
    
    val_dataset = LoTSSDataset(
        args.data_path, split='val', transform=transform,
        val_split=0.1, seed=args.seed
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
        persistent_workers=num_workers > 0 and not is_windows
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0 and not is_windows
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Fixed validation batch
    fixed_batch = next(iter(val_loader))
    fixed_images = fixed_batch[0][:4].to(device)
    if args.mode == 'radio2optical':
        fixed_input = fixed_images[:, 1:2, :, :]
        fixed_target = fixed_images[:, 0:1, :, :]
    else:
        fixed_input = fixed_images[:, 0:1, :, :]
        fixed_target = fixed_images[:, 1:2, :, :]
    
    # Models - U-Net Generator + PatchGAN Discriminator
    G = UNetGenerator(in_channels=1, out_channels=1, ngf=args.ngf).to(device)
    D = PatchGANDiscriminator(in_channels=2, ndf=args.ndf).to(device)
    
    g_params = G.get_num_params()
    d_params = D.get_num_params()
    print(f"\nGenerator params: {g_params['total']:,}")
    print(f"Discriminator params: {d_params['total']:,}")
    print(f"Total params: {g_params['total'] + d_params['total']:,}\n")
    
    # Optimizers - Adam with beta1=0.5 (standard for GANs)
    g_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # Schedulers
    g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=args.num_epochs, eta_min=1e-6)
    d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=args.num_epochs, eta_min=1e-6)
    
    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_L1 = torch.nn.L1Loss()
    vgg_loss = VGGLoss().to(device)
    
    # Mixed precision
    use_amp = args.use_amp and device.type == 'cuda'
    g_scaler = GradScaler('cuda') if use_amp else None
    d_scaler = GradScaler('cuda') if use_amp else None
    
    # Metrics
    tracker = GANMetricsTracker(output_dir)
    
    clear_memory()
    
    print("=" * 60)
    print(f"TRAINING START - {args.num_epochs} Epochs")
    print("=" * 60 + "\n")
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        G.train()
        D.train()
        
        g_loss_sum = {'total': 0, 'adv': 0, 'l1': 0, 'vgg': 0}
        d_loss_sum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        
        for batch_idx, (images, _, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            
            if args.mode == 'radio2optical':
                input_img = images[:, 1:2, :, :]
                target_img = images[:, 0:1, :, :]
            else:
                input_img = images[:, 0:1, :, :]
                target_img = images[:, 1:2, :, :]
            
            # Normalize to [-1, 1]
            input_img = input_img * 2 - 1
            target_img = target_img * 2 - 1
            
            batch_size = input_img.size(0)
            real_label = torch.ones(batch_size, 1, 73, 73, device=device)
            fake_label = torch.zeros(batch_size, 1, 73, 73, device=device)
            
            # ================== Train Discriminator ==================
            d_opt.zero_grad(set_to_none=True)
            
            if use_amp:
                with autocast('cuda'):
                    # Generate fake image
                    fake_img = G(input_img)
                    
                    # Real loss
                    pred_real = D(input_img, target_img)
                    real_label_sized = F.interpolate(real_label, size=pred_real.shape[2:], mode='nearest')
                    loss_D_real = criterion_GAN(pred_real, real_label_sized)
                    
                    # Fake loss
                    pred_fake = D(input_img, fake_img.detach())
                    fake_label_sized = F.interpolate(fake_label, size=pred_fake.shape[2:], mode='nearest')
                    loss_D_fake = criterion_GAN(pred_fake, fake_label_sized)
                    
                    d_loss = (loss_D_real + loss_D_fake) * 0.5
                    d_loss = d_loss / args.grad_accum
                
                d_scaler.scale(d_loss).backward()
                if (batch_idx + 1) % args.grad_accum == 0:
                    d_scaler.step(d_opt)
                    d_scaler.update()
            else:
                fake_img = G(input_img)
                pred_real = D(input_img, target_img)
                real_label_sized = F.interpolate(real_label, size=pred_real.shape[2:], mode='nearest')
                loss_D_real = criterion_GAN(pred_real, real_label_sized)
                
                pred_fake = D(input_img, fake_img.detach())
                fake_label_sized = F.interpolate(fake_label, size=pred_fake.shape[2:], mode='nearest')
                loss_D_fake = criterion_GAN(pred_fake, fake_label_sized)
                
                d_loss = (loss_D_real + loss_D_fake) * 0.5
                d_loss = d_loss / args.grad_accum
                d_loss.backward()
                
                if (batch_idx + 1) % args.grad_accum == 0:
                    d_opt.step()
            
            # ================== Train Generator ==================
            g_opt.zero_grad(set_to_none=True)
            
            if use_amp:
                with autocast('cuda'):
                    fake_img = G(input_img)
                    pred_fake = D(input_img, fake_img)
                    
                    # Adversarial loss
                    real_label_sized = F.interpolate(real_label, size=pred_fake.shape[2:], mode='nearest')
                    g_adv_loss = criterion_GAN(pred_fake, real_label_sized)
                    
                    # L1 loss
                    g_l1_loss = criterion_L1(fake_img, target_img)
                    
                    # VGG loss
                    g_vgg_loss = vgg_loss(fake_img, target_img)
                    
                    # Total
                    g_loss = g_adv_loss + args.lambda_l1 * g_l1_loss + args.lambda_vgg * g_vgg_loss
                    g_loss = g_loss / args.grad_accum
                
                g_scaler.scale(g_loss).backward()
                if (batch_idx + 1) % args.grad_accum == 0:
                    g_scaler.step(g_opt)
                    g_scaler.update()
            else:
                fake_img = G(input_img)
                pred_fake = D(input_img, fake_img)
                
                real_label_sized = F.interpolate(real_label, size=pred_fake.shape[2:], mode='nearest')
                g_adv_loss = criterion_GAN(pred_fake, real_label_sized)
                g_l1_loss = criterion_L1(fake_img, target_img)
                g_vgg_loss = vgg_loss(fake_img, target_img)
                
                g_loss = g_adv_loss + args.lambda_l1 * g_l1_loss + args.lambda_vgg * g_vgg_loss
                g_loss = g_loss / args.grad_accum
                g_loss.backward()
                
                if (batch_idx + 1) % args.grad_accum == 0:
                    g_opt.step()
            
            # Accumulate losses
            g_loss_sum['total'] += g_loss.item() * args.grad_accum
            g_loss_sum['adv'] += g_adv_loss.item()
            g_loss_sum['l1'] += g_l1_loss.item()
            g_loss_sum['vgg'] += g_vgg_loss.item()
            d_loss_sum += d_loss.item() * args.grad_accum
            
            pbar.set_postfix({
                'G': f"{g_loss.item() * args.grad_accum:.2f}",
                'D': f"{d_loss.item() * args.grad_accum:.3f}",
                'L1': f"{g_l1_loss.item():.3f}"
            })
        
        # Epoch summary
        n = len(train_loader)
        g_loss_avg = {k: v / n for k, v in g_loss_sum.items()}
        d_loss_avg = d_loss_sum / n
        
        g_sched.step()
        d_sched.step()
        
        tracker.update(g_loss_avg, d_loss_avg,
                      g_opt.param_groups[0]['lr'], d_opt.param_groups[0]['lr'])
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs} Summary")
        print(f"{'='*60}")
        print(f"  G Loss: {g_loss_avg['total']:.4f} (Adv: {g_loss_avg['adv']:.4f}, "
              f"L1: {g_loss_avg['l1']:.4f}, VGG: {g_loss_avg['vgg']:.4f})")
        print(f"  D Loss: {d_loss_avg:.4f}")
        print(f"  LR: G={g_opt.param_groups[0]['lr']:.2e}, D={d_opt.param_groups[0]['lr']:.2e}")
        
        # Validation
        if epoch % args.save_interval == 0:
            G.eval()
            with torch.no_grad():
                fixed_input_norm = fixed_input * 2 - 1
                fixed_target_norm = fixed_target * 2 - 1
                
                if use_amp:
                    with autocast('cuda'):
                        fake_fixed = G(fixed_input_norm)
                else:
                    fake_fixed = G(fixed_input_norm)
                
                val_psnr, val_ssim = compute_metrics(fake_fixed, fixed_target_norm)
            
            is_best = tracker.add_image_metrics(val_psnr, val_ssim, epoch)
            print(f"  Validation: PSNR={val_psnr:.2f}dB, SSIM={val_ssim:.4f}")
            
            save_samples(fixed_input_norm, fake_fixed, fixed_target_norm, epoch,
                        output_dir / 'samples', args.mode)
            
            if is_best:
                torch.save({
                    'generator': G.state_dict(),
                    'discriminator': D.state_dict(),
                    'epoch': epoch,
                    'psnr': val_psnr,
                    'ssim': val_ssim,
                    'config': config
                }, output_dir / 'best_model.pth')
                print(f"  >>> NEW BEST MODEL! PSNR: {val_psnr:.2f}dB")
            
            clear_memory()
        
        if epoch % 20 == 0:
            torch.save({
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
                'epoch': epoch,
                'config': config
            }, output_dir / 'checkpoints' / f'checkpoint_{epoch}.pth')
        
        print(f"{'='*60}\n")
    
    # Final saves
    torch.save(G.state_dict(), output_dir / 'generator_final.pth')
    torch.save(D.state_dict(), output_dir / 'discriminator_final.pth')
    tracker.plot_curves()
    tracker.save_metrics()
    
    print("\n" + "=" * 60)
    print("GAN TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best PSNR: {tracker.best_psnr:.2f}dB (Epoch {tracker.best_epoch})")
    print(f"  Best SSIM: {max(tracker.ssim_scores) if tracker.ssim_scores else 0:.4f}")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()