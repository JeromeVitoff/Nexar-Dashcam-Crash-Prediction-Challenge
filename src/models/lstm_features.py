"""
LSTM Classifier (Features Only)
================================

LSTM simple qui travaille sur features pré-extraites.
Beaucoup plus rapide que ResNet-LSTM complet.

Auteur: Jerome
Date: Janvier 2026
"""

import torch
import torch.nn as nn


class LSTMFeatureClassifier(nn.Module):
    """
    LSTM qui travaille directement sur features pré-extraites.
    
    Args:
        input_dim: Dimension des features (2048 pour ResNet50)
        hidden_dim: Hidden dimension du LSTM
        num_layers: Nombre de couches LSTM
        num_classes: Nombre de classes (2 pour binary)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
        
        # Stats
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"✅ LSTMFeatureClassifier initialisé:")
        print(f"   • Input dim: {input_dim}")
        print(f"   • LSTM: {num_layers} layers, hidden={hidden_dim}")
        print(f"   • Bidirectional: {bidirectional}")
        print(f"   • Dropout: {dropout}")
        print(f"   • Num classes: {num_classes}")
        print(f"   • Paramètres totaux: {total_params:,}")
        print(f"   • Paramètres entraînables: {trainable_params:,}")
    
    def _init_weights(self):
        """Initialisation des poids."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, num_frames, 2048) - Features pré-extraites
            
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        # lstm_out: (batch, num_frames, hidden_dim)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        
        # Utiliser la dernière sortie
        if self.bidirectional:
            # Concaténer forward et backward du dernier layer
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(last_hidden)
        
        return logits


def test_model():
    """Test du modèle avec données factices."""
    print("🧪 TEST DU MODÈLE\n")
    print("="*70)
    
    # Créer modèle
    model = LSTMFeatureClassifier(
        input_dim=2048,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        bidirectional=False
    )
    
    # Données factices
    batch_size = 8
    num_frames = 16
    features = torch.randn(batch_size, num_frames, 2048)
    
    print(f"\n📥 Input:")
    print(f"   • Shape: {features.shape}")
    print(f"   • (batch, num_frames, feature_dim)")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(features)
    
    print(f"\n📤 Output:")
    print(f"   • Shape: {output.shape}")
    print(f"   • (batch, num_classes)")
    
    print("\n" + "="*70)
    print("✅ TEST RÉUSSI!")
    print("="*70)


if __name__ == "__main__":
    test_model()
