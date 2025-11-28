"""
ResNet-34 for LoTSS Classification - Network-safe version
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34_LoTSS(nn.Module):
    """ResNet-34 adapted for LoTSS dataset"""
    
    def __init__(self, num_classes=6, pretrained=True, dropout=0.5):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet-34 (handle network errors gracefully)
        try:
            if pretrained:
                print("  Attempting to load pretrained weights...")
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                print("  ✓ Loaded pretrained weights")
            else:
                resnet = models.resnet34(weights=None)
                print("  Training from scratch (no pretrained weights)")
        except Exception as e:
            print(f"  ⚠️  Could not load pretrained weights: {e}")
            print("  Training from scratch instead...")
            resnet = models.resnet34(weights=None)
            pretrained = False
        
        # Modify first conv: 2 channels → 64 features
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize conv1
        if pretrained:
            pretrained_weight = resnet.conv1.weight.data
            # Average RGB channels for initialization
            self.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Copy other layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Custom classifier for 6 classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == "__main__":
    model = ResNet34_LoTSS(num_classes=6, pretrained=True)
    params = model.get_num_params()
    
    print(f"ResNet-34 LoTSS")
    print(f"  Parameters: {params['total']:,} (trainable: {params['trainable']:,})")
    
    x = torch.randn(2, 2, 600, 600)
    output = model(x)
    print(f"  Input: {x.shape} → Output: {output.shape}")
