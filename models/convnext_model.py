"""
ConvNeXt-Tiny for LoTSS
Modern CNN architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtTiny_LoTSS(nn.Module):
    """
    ConvNeXt-Tiny adapted for LoTSS
    """
    
    def __init__(self, num_classes=6, pretrained=True, dropout=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ConvNeXt-Tiny
        if pretrained:
            convnext = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        else:
            convnext = models.convnext_tiny(weights=None)
        
        self.stem = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=4, stride=4),
            nn.LayerNorm([96, 150, 150])  # 600/4 = 150
        )
        
        if pretrained:
            # Initialize with pretrained weights (first 2 channels)
            pretrained_weight = convnext.features[0][0].weight.data
            self.stem[0].weight.data = pretrained_weight[:, :2, :, :].clone()
        else:
            nn.init.kaiming_normal_(self.stem[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Copy rest of features (skip stem)
        self.features = nn.Sequential(
            self.stem,
            *list(convnext.features.children())[1:]
        )
        
        self.avgpool = convnext.avgpool
        
        # Custom classifier
        in_features = 768  # ConvNeXt-Tiny output dim
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 384),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(384, num_classes)
        )
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == "__main__":
    model = ConvNeXtTiny_LoTSS(num_classes=6)
    params = model.get_num_params()
    
    print(f"ConvNeXt-Tiny LoTSS")
    print(f"  Parameters: {params['total']:,} (trainable: {params['trainable']:,})")
    
    x = torch.randn(2, 2, 600, 600)
    output = model(x)
    print(f"  Input: {x.shape} → Output: {output.shape}")
