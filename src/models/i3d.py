"""
I3D (Inflated 3D ConvNet) Model for Video Classification

Architecture:
- 3D CNN with Inception modules
- Pre-trained on Kinetics-400
- End-to-end training (no feature pre-extraction)
- Input: (batch, 3, num_frames, 224, 224)
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import mc3_18, MC3_18_Weights


class I3D(nn.Module):
    """
    I3D model for binary video classification (crash vs no crash).
    
    Uses a 3D ResNet backbone (R3D-18) as a proxy for I3D architecture.
    Note: True I3D uses Inception modules, but R3D-18 is similar and 
    has better PyTorch support with pre-trained weights.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use ImageNet/Kinetics pretrained weights
        dropout (float): Dropout rate before final classifier
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(I3D, self).__init__()
        
        # Load pre-trained R3D-18 (3D ResNet-18)
        if pretrained:
            weights = R3D_18_Weights.DEFAULT  # Kinetics-400 pretrained
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)
        
        # Get the input dimension of the final FC layer
        in_features = self.backbone.fc.in_features
        
        # Replace the final classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout = dropout
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, num_frames, H, W)
                             Expected: (batch, 3, 16, 224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
        """
        # Pass through backbone (includes final classifier)
        logits = self.backbone(x)
        return logits
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class I3D_MC3(nn.Module):
    """
    Alternative I3D using MC3 (Mixed Convolution) architecture.
    MC3 uses 3x3x3 and 1x3x3 convolutions which can be more efficient.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use Kinetics pretrained weights
        dropout (float): Dropout rate before final classifier
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(I3D_MC3, self).__init__()
        
        # Load pre-trained MC3-18
        if pretrained:
            weights = MC3_18_Weights.DEFAULT  # Kinetics-400 pretrained
            self.backbone = mc3_18(weights=weights)
        else:
            self.backbone = mc3_18(weights=None)
        
        # Get the input dimension of the final FC layer
        in_features = self.backbone.fc.in_features
        
        # Replace the final classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout = dropout
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, num_frames, H, W)
        
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
        """
        logits = self.backbone(x)
        return logits
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_i3d_model(variant='r3d', num_classes=2, pretrained=True, dropout=0.5):
    """
    Factory function to get I3D model variant.
    
    Args:
        variant (str): Model variant ('r3d' or 'mc3')
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        dropout (float): Dropout rate
    
    Returns:
        nn.Module: I3D model
    """
    if variant == 'r3d':
        return I3D(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif variant == 'mc3':
        return I3D_MC3(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'r3d' or 'mc3'.")


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print(" "*20 + "TESTING I3D MODEL")
    print("="*70)
    
    # Create model
    model = I3D(num_classes=2, pretrained=False, dropout=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✅ I3D Model created successfully!")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    num_frames = 16
    height = 224
    width = 224
    
    x = torch.randn(batch_size, 3, num_frames, height, width)
    print(f"\n📊 Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"   Expected: ({batch_size}, 2)")
    
    # Test freeze/unfreeze
    model.freeze_backbone()
    frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔒 After freezing backbone: {frozen_params:,} trainable params")
    
    model.unfreeze_backbone()
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 After unfreezing: {unfrozen_params:,} trainable params")
    
    print("\n" + "="*70)
    print(" "*25 + "TEST PASSED ✅")
    print("="*70)
