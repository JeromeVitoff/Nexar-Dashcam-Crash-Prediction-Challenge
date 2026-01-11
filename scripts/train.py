"""
Script d'Entraînement Principal
================================

Lance l'entraînement pour les différents modèles.

Usage:
    python scripts/train.py --model resnet_lstm --num_frames 16 --epochs 50
    python scripts/train.py --config config/exp_001.yaml

Auteur: Jerome
Date: Octobre 2025
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import create_trainer
from src.data.dataloader import get_dataloaders

# Import des modèles
from src.models.resnet_lstm import ResNetLSTM
from src.models.efficientnet_gru import EfficientNetGRU
from src.models.i3d import I3D
from src.models.r2plus1d import R2Plus1D
from src.models.timesformer import TimeSformerClassifier
from src.models.videomae import VideoMAEClassifier
from src.models.vivit import ViViTClassifier


def create_model(model_name: str, num_classes: int = 2):
    """
    Crée le modèle selon le nom.
    
    Args:
        model_name: Nom du modèle
        num_classes: Nombre de classes
        
    Returns:
        Modèle PyTorch
    """
    print(f"\n🏗️  Création du modèle: {model_name}")
    
    if model_name == 'resnet_lstm':
        model = ResNetLSTM(
            num_classes=num_classes,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            dropout=0.3,
            freeze_backbone=False,
            pretrained=True
        )
    
    elif model_name == 'efficientnet_gru':
        model = EfficientNetGRU(
            num_classes=num_classes,
            gru_hidden_size=256,
            gru_num_layers=2,
            dropout=0.3,
            freeze_backbone=False,
            pretrained=True
        )
    
    elif model_name == 'i3d':
        model = I3D(
            num_classes=num_classes,
            dropout=0.5
        )
    
    elif model_name == 'r2plus1d':
        model = R2Plus1D(
            num_classes=num_classes,
            dropout=0.5,
            layer_sizes=[2, 2, 2, 2]
        )
    
    elif model_name == 'timesformer':
        model = TimeSformerClassifier(
            num_classes=num_classes,
            pretrained_model="facebook/timesformer-base-finetuned-k400",
            freeze_backbone=False,
            dropout=0.3
        )
    
    elif model_name == 'videomae':
        model = VideoMAEClassifier(
            num_classes=num_classes,
            pretrained_model="MCG-NJU/videomae-base-finetuned-kinetics",
            freeze_backbone=False,
            dropout=0.3
        )
    
    elif model_name == 'vivit':
        model = ViViTClassifier(
            num_classes=num_classes,
            pretrained_model="google/vivit-b-16x2-kinetics400",
            freeze_backbone=False,
            dropout=0.3
        )
    
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   • Paramètres totaux: {total_params:,}")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    
    return model


def permute_for_3d_models(frames, model_name):
    """
    Permute les dimensions pour les modèles 3D.
    
    2D models: (B, T, C, H, W)
    3D models: (B, C, T, H, W)
    """
    if model_name in ['i3d', 'r2plus1d']:
        return frames.permute(0, 2, 1, 3, 4)
    return frames


class ModelWrapper(torch.nn.Module):
    """Wrapper pour gérer les formats d'entrée différents."""
    
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name
    
    def forward(self, x):
        # Permuter si modèle 3D
        if self.model_name in ['i3d', 'r2plus1d']:
            x = x.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W) → (B,C,T,H,W)
        return self.model(x)


def validate_config(config, model_name):
    """Valide la configuration selon le modèle."""
    num_frames = config['num_frames']
    
    # VideoMAE nécessite 16 frames
    if model_name == 'videomae' and num_frames != 16:
        print(f"⚠️  VideoMAE nécessite 16 frames (config: {num_frames})")
        print(f"   → Ajustement automatique à 16 frames")
        config['num_frames'] = 16
    
    # ViViT nécessite 32 frames
    if model_name == 'vivit' and num_frames != 32:
        print(f"⚠️  ViViT nécessite 32 frames (config: {num_frames})")
        print(f"   → Ajustement automatique à 32 frames")
        config['num_frames'] = 32
    
    return config


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Entraînement de modèles de classification vidéo")
    
    # Arguments
    parser.add_argument('--config', type=str, help='Fichier config YAML')
    parser.add_argument('--model', type=str, 
                       choices=['resnet_lstm', 'efficientnet_gru', 'i3d', 'r2plus1d', 
                               'timesformer', 'videomae', 'vivit'],
                       help='Nom du modèle')
    parser.add_argument('--num_frames', type=int, default=16, help='Nombre de frames')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train_csv', type=str, default='Data/train.csv')
    parser.add_argument('--train_dir', type=str, default='Data/train')
    
    args = parser.parse_args()
    
    # Charger config depuis YAML ou arguments
    if args.config:
        print(f"📄 Chargement config: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config['model']
    else:
        if not args.model:
            parser.error("--model requis si pas de --config")
        model_name = args.model
        config = {
            'model': model_name,
            'num_frames': args.num_frames,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'scheduler': 'reduce_on_plateau',
            'patience': 10
        }
    
    # Valider config selon le modèle
    config = validate_config(config, model_name)
    
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT DE MODÈLE")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   • Modèle: {model_name}")
    print(f"   • Frames: {config['num_frames']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Epochs: {config['epochs']}")
    print(f"   • Learning rate: {config['lr']}")
    
    # 1. Charger les données
    print(f"\n📥 Chargement des données...")
    train_loader, val_loader = get_dataloaders(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        val_split=0.2,
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        sampling_strategy='uniform',
        augmentation_level='basic',
        num_workers=2,
        seed=42
    )
    
    print(f"   ✅ Train: {len(train_loader)} batches")
    print(f"   ✅ Val: {len(val_loader)} batches")
    
    # 2. Créer le modèle
    model = create_model(model_name, num_classes=2)
    
    # Wrapper pour gérer les formats
    wrapped_model = ModelWrapper(model, model_name)
    
    # 3. Créer save directory
    save_dir = Path('checkpoints') / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config['save_dir'] = str(save_dir)
    
    # 4. Créer le Trainer
    print(f"\n⚙️  Création du Trainer...")
    trainer = create_trainer(
        model=wrapped_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # 5. LANCER l'entraînement
    print(f"\n{'='*70}")
    print(f"🎯 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*70}\n")
    
    try:
        trainer.train(num_epochs=config['epochs'])
        
        print(f"\n{'='*70}")
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"{'='*70}")
        print(f"\n📁 Résultats sauvegardés dans: {save_dir}")
        print(f"   • Meilleur modèle: {save_dir}/best_model.pth")
        print(f"   • Métriques: {save_dir}/metrics.json")
        print(f"   • Log: {save_dir}/training.log")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur")
        print(f"   Checkpoints sauvegardés dans: {save_dir}")
    
    except Exception as e:
        print(f"\n\n❌ ERREUR pendant l'entraînement:")
        print(f"   {str(e)}")
        raise


if __name__ == "__main__":
    main()
