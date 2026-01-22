"""
Modèle VideoMAE pour la classification vidéo
Vision Transformer avec Masked Autoencoding pré-entraîné
"""

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

class VideoMAEModel(nn.Module):
    def __init__(self, num_frames=8, img_size=224, num_classes=2, pretrained=True):
        super().__init__()
        
        self.num_frames = num_frames
        self.img_size = img_size
        
        if pretrained:
            # Charger VideoMAE pré-entraîné sur Kinetics-400
            self.model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            print("✅ VideoMAE chargé avec pré-entraînement Kinetics-400")
        else:
            # Version from scratch (non recommandé après l'échec de TimeSformer)
            from transformers import VideoMAEConfig
            config = VideoMAEConfig(
                num_labels=num_classes,
                image_size=img_size,
                num_frames=num_frames
            )
            self.model = VideoMAEForVideoClassification(config)
            print("⚠️ VideoMAE sans pré-entraînement (risqué)")
    
    def forward(self, x):
        """
        Args:
            x: VideoDataset retourne (B, T, C, H, W) après batching
               où B=batch, T=num_frames, C=channels, H=height, W=width
        Returns:
            logits: (batch, num_classes)
        """
        # ✅ PAS DE PERMUTE ! Les données arrivent déjà au bon format
        # x shape: (batch, 8, 3, 160, 160) = (B, T, C, H, W)
        
        # Vérification du format
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D tensor (B, T, C, H, W), got shape {x.shape}")
        
        batch_size, num_frames, num_channels, height, width = x.shape
        
        # VideoMAE attend des tenseurs avec channels=3
        if num_channels != 3:
            raise ValueError(f"VideoMAE expects 3 channels (RGB), got {num_channels}")
        
        # Forward pass - VideoMAE attend pixel_values de shape (B, T, C, H, W)
        outputs = self.model(pixel_values=x)
        return outputs.logits
