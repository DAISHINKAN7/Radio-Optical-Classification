"""
EfficientNet-B0 for LoTSS Classification
Modified for 2-channel input and 6 classes
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB0_LoTSS(nn.Module):
    """
    EfficientNet-B0 adapted for LoTSS
    """
    
    def __init__(self, num_classes=6, pretrained=True, dropout=0.5):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv: 2 channels → 32 features
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.first_conv = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        if pretrained:
            # Use first 2 channels of pretrained weights
            pretrained_weight = efficientnet.features[0][0].weight.data
            self.first_conv.weight.data = pretrained_weight[:, :2, :, :].clone()
        else:
            nn.init.kaiming_normal_(self.first_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Copy rest of features (skip first conv)
        self.features = nn.Sequential(
            self.first_conv,
            *list(efficientnet.features.children())[1:]
        )
        
        self.avgpool = efficientnet.avgpool
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == "__main__":
    model = EfficientNetB0_LoTSS(num_classes=6)
    params = model.get_num_params()
    
    print(f"EfficientNet-B0 LoTSS")
    print(f"  Parameters: {params['total']:,} (trainable: {params['trainable']:,})")
    
    x = torch.randn(2, 2, 600, 600)
    output = model(x)
    print(f"  Input: {x.shape} → Output: {output.shape}")
