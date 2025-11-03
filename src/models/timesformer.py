"""
TimeSformer (Hugging Face) pour Classification de Vidéos
=========================================================

Modèle Transformer pré-entraîné pour vidéos (Facebook AI).

Source: "Is Space-Time Attention All You Need for Video Understanding?" (2021)
Hugging Face: facebook/timesformer-base-finetuned-k400

Architecture:
    - Vision Transformer adapté pour vidéos
    - Attention spatio-temporelle divisée
    - Pré-entraîné sur Kinetics-400

Auteur: Jerome
Expérience: exp_005 (Transformer baseline)
"""

import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig
from typing import Tuple, Dict


class TimeSformerClassifier(nn.Module):
    """
    TimeSformer pour classification de collisions.
    
    Utilise un modèle pré-entraîné de Hugging Face et ajoute
    une tête de classification custom.
    
    Args:
        num_classes (int): Nombre de classes (2)
        pretrained_model (str): Nom du modèle HF
        freeze_backbone (bool): Freeze le transformer
        dropout (float): Dropout pour le classifier
        
    Input Shape:
        (batch, num_frames, 3, 224, 224)
        
    Output Shape:
        (batch, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained_model: str = "facebook/timesformer-base-finetuned-k400",
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(TimeSformerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self._is_backbone_frozen = freeze_backbone
        
        # Charger le modèle pré-entraîné
        print(f"   📥 Chargement de {pretrained_model}...")
        self.timesformer = TimesformerModel.from_pretrained(pretrained_model)
        
        # Dimension des features
        self.hidden_size = self.timesformer.config.hidden_size  # 768
        
        # Freeze si demandé
        if freeze_backbone:
            for param in self.timesformer.parameters():
                param.requires_grad = False
            print("   🔒 Backbone freezé")
        else:
            print("   🔓 Backbone entraînable")
        
        # Classifier custom
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        print(f"✅ TimeSformer initialisé:")
        print(f"   • Model: {pretrained_model}")
        print(f"   • Hidden size: {self.hidden_size}")
        print(f"   • Dropout: {dropout}")
        print(f"   • Num classes: {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, num_frames, 3, H, W)
            
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]
        
        # TimeSformer attend (batch, num_frames, channels, H, W)
        # C'est déjà le bon format !
        
        # Forward à travers TimeSformer
        outputs = self.timesformer(x)
        
        # Prendre le [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]  # (batch, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_num_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def freeze_backbone(self):
        for param in self.timesformer.parameters():
            param.requires_grad = False
        self._is_backbone_frozen = True
        print("🔒 Backbone freezé")
    
    def unfreeze_backbone(self):
        for param in self.timesformer.parameters():
            param.requires_grad = True
        self._is_backbone_frozen = False
        print("🔓 Backbone unfreezé")
    
    def unfreeze_last_n_layers(self, n: int = 2):
        """Unfreeze les n dernières couches du transformer."""
        self.freeze_backbone()
        
        # TimeSformer a 12 layers
        total_layers = len(self.timesformer.encoder.layer)
        start_idx = max(0, total_layers - n)
        
        for i in range(start_idx, total_layers):
            for param in self.timesformer.encoder.layer[i].parameters():
                param.requires_grad = True
        
        print(f"🔓 Dernières {n} couches unfreezées")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_percent': (trainable / total * 100) if total > 0 else 0
    }


def test_model():
    """Test du modèle TimeSformer."""
    print("🧪 TEST DU MODÈLE TimeSformer\n")
    print("="*70)
    
    # 1. Créer le modèle
    print("\n1️⃣ Création du modèle...")
    model = TimeSformerClassifier(
        num_classes=2,
        pretrained_model="facebook/timesformer-base-finetuned-k400",
        freeze_backbone=False,
        dropout=0.3
    )
    
    # 2. Statistiques
    print("\n2️⃣ Statistiques:")
    params = count_parameters(model)
    print(f"   • Total: {params['total']:,}")
    print(f"   • Trainable: {params['trainable']:,}")
    print(f"   • Frozen: {params['frozen']:,}")
    
    # 3. Test forward
    print("\n3️⃣ Test forward pass...")
    batch_size = 2
    num_frames = 8  # TimeSformer peut prendre 8 frames
    x = torch.randn(batch_size, num_frames, 3, 224, 224)
    print(f"   • Input: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   • Output: {output.shape}")
    print(f"   • Range: [{output.min():.3f}, {output.max():.3f}]")
    
    # 4. Test configurations
    print("\n4️⃣ Test configurations...")
    for nf in [8, 16]:
        x = torch.randn(1, nf, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        print(f"   ✓ {nf} frames: {out.shape}")
    
    # 5. GPU
    print("\n5️⃣ Test GPU...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"   • Device: {device}")
    
    try:
        model = model.to(device)
        x = torch.randn(1, 8, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"   ✓ Forward sur {device}: OK")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 6. Freeze/Unfreeze
    print("\n6️⃣ Test freeze/unfreeze...")
    p_before = count_parameters(model)['trainable']
    print(f"   • Avant: {p_before:,}")
    
    model.freeze_backbone()
    p_frozen = count_parameters(model)['trainable']
    print(f"   • Après freeze: {p_frozen:,}")
    
    model.unfreeze_last_n_layers(2)
    p_partial = count_parameters(model)['trainable']
    print(f"   • Unfreeze 2 layers: {p_partial:,}")
    
    print("\n" + "="*70)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*70)
    
    # Résumé
    print("\n📊 RÉSUMÉ:")
    print("-" * 70)
    print(f"Architecture: TimeSformer (Vision Transformer)")
    print(f"Paramètres: {params['total']:,}")
    print(f"Source: facebook/timesformer-base-finetuned-k400")
    print(f"Pré-entraîné: Kinetics-400")
    print(f"Type: Transformer avec attention spatio-temporelle")
    print("-" * 70)


if __name__ == "__main__":
    test_model()
