"""
Pix2Pix GAN for Radio <-> Optical Translation
U-Net Generator (better for paired image translation)
Optimized for 16GB GPU with 600x600 images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================================
# U-NET GENERATOR (Better for Image-to-Image Translation)
# ============================================================================

class UNetDown(nn.Module):
    """U-Net Encoder Block"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net Decoder Block"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for 600x600 images
    Takes input image directly (not random noise)
    """
    def __init__(self, in_channels=1, out_channels=1, ngf=64):
        super().__init__()
        
        # Encoder (downsampling): 600 -> 300 -> 150 -> 75 -> 37 -> 18 -> 9 -> 4
        self.down1 = UNetDown(in_channels, ngf, normalize=False)  # 300
        self.down2 = UNetDown(ngf, ngf * 2)                        # 150
        self.down3 = UNetDown(ngf * 2, ngf * 4)                    # 75
        self.down4 = UNetDown(ngf * 4, ngf * 8)                    # 37
        self.down5 = UNetDown(ngf * 8, ngf * 8)                    # 18
        self.down6 = UNetDown(ngf * 8, ngf * 8)                    # 9
        self.down7 = UNetDown(ngf * 8, ngf * 8, normalize=False)   # 4
        
        # Decoder (upsampling with skip connections)
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)           # 9
        self.up2 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)          # 18
        self.up3 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)          # 37
        self.up4 = UNetUp(ngf * 16, ngf * 4)                       # 75
        self.up5 = UNetUp(ngf * 8, ngf * 2)                        # 150
        self.up6 = UNetUp(ngf * 4, ngf)                            # 300
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)    # 300
        d2 = self.down2(d1)   # 150
        d3 = self.down3(d2)   # 75
        d4 = self.down4(d3)   # 37
        d5 = self.down5(d4)   # 18
        d6 = self.down6(d5)   # 9
        d7 = self.down7(d6)   # 4
        
        # Decoder with skip connections
        u1 = self.up1(d7, d6)  # 9
        u2 = self.up2(u1, d5)  # 18
        u3 = self.up3(u2, d4)  # 37
        u4 = self.up4(u3, d3)  # 75
        u5 = self.up5(u4, d2)  # 150
        u6 = self.up6(u5, d1)  # 300
        
        # Final output
        out = self.final(u6)  # 600
        
        # Ensure exact 600x600 output
        if out.shape[2:] != (600, 600):
            out = F.interpolate(out, size=(600, 600), mode='bilinear', align_corners=False)
        
        return out
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ============================================================================
# PATCHGAN DISCRIMINATOR
# ============================================================================

class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator - 70x70 receptive field"""
    def __init__(self, in_channels=2, ndf=64, n_layers=3):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Final layers
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ============================================================================
# VGG PERCEPTUAL LOSS
# ============================================================================

class VGGLoss(nn.Module):
    """VGG Perceptual Loss - Memory Optimized"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(2)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7, 12)])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Resize for memory efficiency
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        x1, y1 = self.slice1(x), self.slice1(y)
        x2, y2 = self.slice2(x1), self.slice2(y1)
        x3, y3 = self.slice3(x2), self.slice3(y2)
        
        loss = F.l1_loss(x1, y1) + F.l1_loss(x2, y2) + F.l1_loss(x3, y3)
        return loss


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Keep old names for compatibility with existing code
SPADEGenerator = UNetGenerator
MultiScaleDiscriminator = PatchGANDiscriminator


if __name__ == "__main__":
    print("Testing Pix2Pix GAN Models (600x600)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = UNetGenerator(in_channels=1, out_channels=1, ngf=64).to(device)
    D = PatchGANDiscriminator(in_channels=2, ndf=64).to(device)
    
    g_params = G.get_num_params()
    d_params = D.get_num_params()
    
    print(f"Generator parameters: {g_params['total']:,}")
    print(f"Discriminator parameters: {d_params['total']:,}")
    print(f"Total parameters: {g_params['total'] + d_params['total']:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    radio = torch.randn(2, 1, 600, 600).to(device)
    
    G.eval()
    with torch.no_grad():
        optical_fake = G(radio)
    
    optical_real = torch.randn(2, 1, 600, 600).to(device)
    disc_out = D(radio, optical_fake)
    
    print(f"Input shape: {radio.shape}")
    print(f"Generated shape: {optical_fake.shape}")
    print(f"Discriminator output: {disc_out.shape}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\n✓ All tests passed!")