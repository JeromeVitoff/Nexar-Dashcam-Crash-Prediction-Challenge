"""
Modèle R(2+1)D pour la classification vidéo
CNN 3D avec décomposition spatiotemporelle
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class R2Plus1DModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Charger R(2+1)D pré-entraîné sur Kinetics
        self.model = r2plus1d_18(pretrained=pretrained)
        
        # Remplacer la dernière couche
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        if pretrained:
            print("✅ R(2+1)D-18 chargé avec pré-entraînement Kinetics")
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, frames, height, width) ou (batch, frames, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        # R(2+1)D attend (batch, channels, frames, height, width)
        if x.shape[1] != 3:  # Si pas au bon format
            # Passer de (B, T, C, H, W) à (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        
        return self.model(x)
