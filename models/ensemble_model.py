"""
Ensemble Model combining EfficientNet-B0 and ConvNeXt-Tiny
Combines predictions for better accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_model import EfficientNetB0_LoTSS
from models.convnext_model import ConvNeXtTiny_LoTSS


class EnsembleClassifier(nn.Module):
    """
    Ensemble of EfficientNet-B0 and ConvNeXt-Tiny
    
    Combination methods:
    - 'average': Average softmax probabilities (default, best for calibrated models)
    - 'vote': Majority voting (good when models disagree)
    - 'weighted': Weighted average based on individual accuracies
    """
    
    def __init__(self, efficientnet_path, convnext_path, num_classes=6, 
                 method='average', weights=None, device='cuda'):
        """
        Args:
            efficientnet_path: Path to EfficientNet checkpoint
            convnext_path: Path to ConvNeXt checkpoint
            num_classes: Number of classes (6)
            method: Ensemble method ('average', 'vote', 'weighted')
            weights: Custom weights [w_eff, w_conv] for weighted average
            device: Device to load models on
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.method = method
        self.device = device
        
        # Load models
        print(f"\nLoading Ensemble Models...")
        print(f"  Method: {method}")
        
        # EfficientNet
        self.efficientnet = EfficientNetB0_LoTSS(num_classes=num_classes, pretrained=False)
        efficientnet_checkpoint = torch.load(efficientnet_path, map_location=device)
        self.efficientnet.load_state_dict(efficientnet_checkpoint['model_state_dict'])
        self.efficientnet.eval()
        print(f"  ✓ Loaded EfficientNet from: {efficientnet_path}")
        print(f"    Val Acc: {efficientnet_checkpoint.get('val_acc', 'N/A'):.2%}")
        
        # ConvNeXt
        self.convnext = ConvNeXtTiny_LoTSS(num_classes=num_classes, pretrained=False)
        convnext_checkpoint = torch.load(convnext_path, map_location=device)
        self.convnext.load_state_dict(convnext_checkpoint['model_state_dict'])
        self.convnext.eval()
        print(f"  ✓ Loaded ConvNeXt from: {convnext_path}")
        print(f"    Val Acc: {convnext_checkpoint.get('val_acc', 'N/A'):.2%}")
        
        # Set ensemble weights
        if weights is None:
            if method == 'weighted':
                # Use validation accuracies as weights
                eff_acc = efficientnet_checkpoint.get('val_acc', 0.8809)
                conv_acc = convnext_checkpoint.get('val_acc', 0.8993)
                total = eff_acc + conv_acc
                self.weights = torch.tensor([eff_acc / total, conv_acc / total], device=device)
                print(f"  Weights (based on val acc): EfficientNet={self.weights[0]:.3f}, ConvNeXt={self.weights[1]:.3f}")
            else:
                self.weights = torch.tensor([0.5, 0.5], device=device)
        else:
            self.weights = torch.tensor(weights, device=device)
            print(f"  Custom weights: {self.weights.tolist()}")
        
        # Move to device
        self.efficientnet = self.efficientnet.to(device)
        self.convnext = self.convnext.to(device)
        
        # Freeze models (for inference)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.convnext.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor (B, 2, 600, 600)
        
        Returns:
            logits: (B, num_classes)
        """
        # Get predictions from both models
        with torch.no_grad():
            eff_logits = self.efficientnet(x)
            conv_logits = self.convnext(x)
        
        if self.method == 'average':
            # Average softmax probabilities
            eff_probs = F.softmax(eff_logits, dim=1)
            conv_probs = F.softmax(conv_logits, dim=1)
            ensemble_probs = (eff_probs + conv_probs) / 2
            # Convert back to logits for loss computation
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.method == 'weighted':
            # Weighted average of probabilities
            eff_probs = F.softmax(eff_logits, dim=1)
            conv_probs = F.softmax(conv_logits, dim=1)
            ensemble_probs = self.weights[0] * eff_probs + self.weights[1] * conv_probs
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.method == 'vote':
            # Majority voting
            eff_preds = torch.argmax(eff_logits, dim=1)
            conv_preds = torch.argmax(conv_logits, dim=1)
            
            # Create one-hot votes
            batch_size = x.size(0)
            votes = torch.zeros(batch_size, self.num_classes, device=self.device)
            votes.scatter_add_(1, eff_preds.unsqueeze(1), torch.ones_like(eff_preds.unsqueeze(1).float()))
            votes.scatter_add_(1, conv_preds.unsqueeze(1), torch.ones_like(conv_preds.unsqueeze(1).float()))
            
            ensemble_logits = votes  # Use vote counts as logits
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return ensemble_logits
    
    def predict(self, x):
        """
        Get predictions and probabilities
        
        Args:
            x: Input tensor (B, 2, 600, 600)
        
        Returns:
            predictions: (B,) class indices
            probabilities: (B, num_classes) probabilities
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs
    
    def get_individual_predictions(self, x):
        """
        Get predictions from individual models for analysis
        
        Returns:
            dict with individual and ensemble predictions
        """
        with torch.no_grad():
            eff_logits = self.efficientnet(x)
            conv_logits = self.convnext(x)
            ensemble_logits = self.forward(x)
            
            eff_probs = F.softmax(eff_logits, dim=1)
            conv_probs = F.softmax(conv_logits, dim=1)
            ensemble_probs = F.softmax(ensemble_logits, dim=1)
            
            return {
                'efficientnet': {
                    'logits': eff_logits,
                    'probs': eff_probs,
                    'preds': torch.argmax(eff_probs, dim=1)
                },
                'convnext': {
                    'logits': conv_logits,
                    'probs': conv_probs,
                    'preds': torch.argmax(conv_probs, dim=1)
                },
                'ensemble': {
                    'logits': ensemble_logits,
                    'probs': ensemble_probs,
                    'preds': torch.argmax(ensemble_probs, dim=1)
                }
            }


if __name__ == "__main__":
    # Test ensemble
    print("Testing Ensemble Model...")
    
    # Dummy paths (update with your actual paths)
    eff_path = "outputs/efficientnet/best_model.pth"
    conv_path = "outputs/convnext/best_model.pth"
    
    if Path(eff_path).exists() and Path(conv_path).exists():
        # Test different ensemble methods
        for method in ['average', 'weighted', 'vote']:
            print(f"\n{'='*60}")
            print(f"Testing: {method.upper()}")
            print(f"{'='*60}")
            
            ensemble = EnsembleClassifier(
                eff_path, conv_path, 
                num_classes=6, 
                method=method,
                device='cpu'
            )
            
            # Test forward pass
            x = torch.randn(2, 2, 600, 600)
            output = ensemble(x)
            print(f"  Input: {x.shape}")
            print(f"  Output: {output.shape}")
            
            # Test prediction
            preds, probs = ensemble.predict(x)
            print(f"  Predictions: {preds}")
            print(f"  Probabilities shape: {probs.shape}")
    else:
        print(f"\n⚠️  Model checkpoints not found:")
        print(f"  EfficientNet: {eff_path}")
        print(f"  ConvNeXt: {conv_path}")
        print(f"\nTrain both models first!")
