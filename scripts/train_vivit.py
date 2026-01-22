"""
Entraînement du modèle ViViT pour la détection de collisions
Architecture : Video Vision Transformer (Factored Encoder)
"""

import sys
sys.path.append('/content/drive/MyDrive/Nexar-Dashcam-Crash-Prediction-Challenge')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.models.vivit import ViViTModel
from src.data.video_dataset import VideoDataset
from src.data.transforms import get_video_transforms
from src.training.trainer import Trainer

def main():
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT VIVIT (Video Vision Transformer - Factored)")
    print("="*70)
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    config = {
        'num_frames': 32,    # ViViT standard utilise 32 frames
        'frame_size': 224,   # Taille standard pour ViT
        'batch_size': 4,     # Petit batch car Transformer gourmand
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'train_split': 0.8,
        'num_workers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'patience': 10,
        'use_amp': True,
        'scheduler': 'reduce_on_plateau'
    }
    
    print("\n📊 Configuration:")
    print(f"   • Model: ViViT (Factored Encoder)")
    print(f"   • Num frames: {config['num_frames']}")
    print(f"   • Frame size: {config['frame_size']}x{config['frame_size']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Epochs: {config['num_epochs']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    
    print(f"\n🖥️  Device: {config['device']}")
    
    # ============================================================
    # CHEMINS
    # ============================================================
    base_dir = Path('/content/drive/MyDrive/Nexar-Dashcam-Crash-Prediction-Challenge')
    video_dir = base_dir / 'Data' / 'train'
    csv_path = base_dir / 'Data' / 'train.csv'
    checkpoint_dir = base_dir / 'checkpoints' / 'vivit'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Chemins:")
    print(f"   • CSV: {csv_path}")
    print(f"   • Vidéos: {video_dir}")
    print(f"   • Checkpoints: {checkpoint_dir}")
    
    # ============================================================
    # CHARGEMENT DES DONNÉES
    # ============================================================
    print("\n📥 Chargement des vidéos...")
    
    transforms = get_video_transforms(
        num_frames=config['num_frames'],
        frame_size=config['frame_size'],
        is_training=True
    )
    
    dataset = VideoDataset(
        csv_file=str(csv_path),
        video_dir=str(video_dir),
        num_frames=config['num_frames'],
        transform=transforms
    )
    
    print(f"✅ VideoDataset initialized with {len(dataset)} videos")
    
    # Split train/val
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n✅ Datasets créés:")
    print(f"   • Train: {len(train_dataset)} vidéos")
    print(f"   • Val: {len(val_dataset)} vidéos")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"\n✅ DataLoaders créés:")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    
    # ============================================================
    # CRÉATION DU MODÈLE
    # ============================================================
    print("\n🏗️  Création du modèle ViViT...")
    
    model = ViViTModel(
        num_frames=config['num_frames'],
        img_size=config['frame_size'],
        num_classes=2,
        pretrained=True  # CRUCIAL après l'échec de TimeSformer
    )
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Modèle créé:")
    print(f"   • Total paramètres: {total_params:,}")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    
    # ============================================================
    # CRÉATION CRITERION ET OPTIMIZER
    # ============================================================
    print("\n⚙️  Configuration de l'entraînement...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    print(f"✅ Configuration créée:")
    print(f"   • Criterion: CrossEntropyLoss")
    print(f"   • Optimizer: AdamW (lr={config['learning_rate']})")
    
    # ============================================================
    # ENTRAÎNEMENT
    # ============================================================
    print("\n⚙️  Création du Trainer...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        save_dir=str(checkpoint_dir)
    )
    
    print("\n🎯 Début de l'entraînement...")
    print("="*70)
    
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("="*70)
    print(f"\n📊 Résultats finaux:")
    print(f"   • Meilleure accuracy: {trainer.best_val_acc:.4f}")

if __name__ == "__main__":
    main()
