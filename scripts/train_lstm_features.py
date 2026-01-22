"""
Entraînement LSTM sur Features Pré-extraites
=============================================

Entraîne le LSTM directement sur les features ResNet50 pré-extraites.
10× plus rapide que l'entraînement complet ResNet-LSTM.

Usage:
    python scripts/train_lstm_features.py --features_dir features/resnet50

Auteur: Jerome
Date: Janvier 2026
"""

import torch
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import create_trainer


def main():
    parser = argparse.ArgumentParser(description="Train LSTM on pre-extracted features")
    
    # Data
    parser.add_argument('--train_csv', type=str, default='Data/train.csv')
    parser.add_argument('--features_dir', type=str, default='features/resnet50')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--bidirectional', action='store_true')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Peut être BEAUCOUP plus grand avec features')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau')
    parser.add_argument('--patience', type=int, default=10)
    
    # Misc
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/lstm_features')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT LSTM (Features Pré-extraites)")
    print("="*70)
    
    print(f"\n📊 Configuration:")
    print(f"   • Features dir: {args.features_dir}")
    print(f"   • Hidden dim: {args.hidden_dim}")
    print(f"   • Num layers: {args.num_layers}")
    print(f"   • Bidirectional: {args.bidirectional}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Learning rate: {args.lr}")
    
    # Import ici pour éviter erreur si pas installé
    from src.models.lstm_features import LSTMFeatureClassifier
    from src.data.feature_dataset import get_feature_dataloaders
    
    # Charger les données
    print(f"\n📥 Chargement des features...")
    train_loader, val_loader = get_feature_dataloaders(
        csv_path=args.train_csv,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    
    # Créer le modèle
    print(f"\n🏗️  Création du modèle...")
    model = LSTMFeatureClassifier(
        input_dim=2048,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    
    # Configuration pour le Trainer
    config = {
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'patience': args.patience,
        'epochs': args.epochs,
        'save_dir': args.save_dir,
        'use_amp': True,  # Mixed precision
        'model_name': 'lstm_features',
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'bidirectional': args.bidirectional
    }
    
    # Créer Trainer
    print(f"\n⚙️  Création du Trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Entraîner
    print(f"\n🎯 Début de l'entraînement...")
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    
    # Résumé
    print(f"\n📊 Résultats finaux:")
    print(f"   • Meilleure accuracy: {trainer.best_val_acc:.4f}")
    print(f"   • Meilleure loss: {trainer.best_val_loss:.4f}")
    print(f"   • Meilleur AP: {max(trainer.val_aps):.4f}")
    
    print(f"\n📁 Résultats sauvegardés dans: {args.save_dir}")
    print(f"   • Meilleur modèle: {args.save_dir}/best_model.pth")
    print(f"   • Métriques: {args.save_dir}/metrics.json")
    print(f"   • Log: {args.save_dir}/training.log")


if __name__ == "__main__":
    main()
