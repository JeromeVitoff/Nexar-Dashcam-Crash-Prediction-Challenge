"""
Modèle ViViT pour la classification vidéo
Video Vision Transformer avec architecture factored
"""

import torch
import torch.nn as nn
from transformers import VivitForVideoClassification, VivitImageProcessor

class ViViTModel(nn.Module):
    def __init__(self, num_frames=8, img_size=224, num_classes=2, pretrained=True):
        super().__init__()
        
        self.num_frames = num_frames
        self.img_size = img_size
        
        if pretrained:
            # Charger ViViT pré-entraîné
            try:
                self.model = VivitForVideoClassification.from_pretrained(
                    "google/vivit-b-16x2-kinetics400",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                print("✅ ViViT chargé avec pré-entraînement Kinetics-400")
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du modèle pré-entraîné: {e}")
                print("⚠️ Utilisation d'un modèle sans pré-entraînement (risqué)")
                from transformers import VivitConfig
                config = VivitConfig(
                    num_labels=num_classes,
                    image_size=img_size,
                    num_frames=num_frames
                )
                self.model = VivitForVideoClassification(config)
        else:
            # Version from scratch (NON RECOMMANDÉ après TimeSformer)
            from transformers import VivitConfig
            config = VivitConfig(
                num_labels=num_classes,
                image_size=img_size,
                num_frames=num_frames
            )
            self.model = VivitForVideoClassification(config)
            print("⚠️ ViViT sans pré-entraînement (risqué après échec TimeSformer)")
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch, num_frames, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        # ViViT attend (batch, num_frames, channels, height, width)
        # C'est déjà le format retourné par VideoDataset
        
        # Vérification du format
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D tensor (B, T, C, H, W), got shape {x.shape}")
        
        batch_size, num_frames, num_channels, height, width = x.shape
        
        # Vérifier que c'est RGB
        if num_channels != 3:
            raise ValueError(f"ViViT expects 3 channels (RGB), got {num_channels}")
        
        # Forward pass
        outputs = self.model(pixel_values=x)
        return outputs.logits
