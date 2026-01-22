"""
Training Script for TimeSformer Model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append('/content/drive/MyDrive/Nexar-Dashcam-Crash-Prediction-Challenge')

from src.models.timesformer import get_timesformer
from src.data.video_dataset import VideoDataset
from src.data.video_transforms import get_video_transforms
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train TimeSformer model')
    
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--frame_size', type=int, default=224)
    parser.add_argument('--sampling', type=str, default='uniform')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/timesformer')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT TIMESFORMER (Vision Transformer)")
    print("="*70)
    
    print(f"\n📊 Configuration:")
    print(f"   • Model: TimeSformer (Divided Space-Time Attention)")
    print(f"   • Num frames: {args.num_frames}")
    print(f"   • Frame size: {args.frame_size}x{args.frame_size}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Epochs: {args.epochs}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Transforms
    train_transform = get_video_transforms(mode='train', img_size=args.frame_size)
    val_transform = get_video_transforms(mode='val', img_size=args.frame_size)
    
    # Dataset
    print(f"\n📥 Chargement des vidéos...")
    full_dataset = VideoDataset(
        csv_file=args.train_csv,
        video_dir=args.train_dir,
        num_frames=args.num_frames,
        transform=train_transform,
        sampling=args.sampling
    )
    
    # Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    
    print(f"\n✅ Datasets créés:")
    print(f"   • Train: {len(train_dataset)} vidéos")
    print(f"   • Val: {len(val_dataset)} vidéos")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"\n✅ DataLoaders créés:")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    
    # Model
    print(f"\n🏗️  Création du modèle TimeSformer...")
    model = get_timesformer(
        num_classes=2, 
        num_frames=args.num_frames, 
        img_size=args.frame_size
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Modèle créé:")
    print(f"   • Total paramètres: {total_params:,}")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # Config for Trainer
    config = {
        'num_epochs': args.epochs,
        'patience': 10,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'save_best_only': True
    }
    
    # Trainer
    print(f"\n⚙️  Création du Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        save_dir=args.save_dir
    )
    
    print(f"✅ Trainer créé:")
    print(f"   • Save dir: {args.save_dir}")
    
    # Train
    print(f"\n🎯 Début de l'entraînement...")
    print("="*70)
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("="*70)
    
    print(f"\n📊 Résultats finaux:")
    print(f"   • Meilleure accuracy: {trainer.best_val_acc:.4f}")

if __name__ == '__main__':
    main()
