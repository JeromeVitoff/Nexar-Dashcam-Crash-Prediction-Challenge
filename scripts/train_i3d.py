"""
Training Script for I3D Model

End-to-end training on full videos (no feature pre-extraction)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.i3d import I3D
from src.data.video_dataset import VideoDataset
from src.data.video_transforms import get_video_transforms
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train I3D model')
    
    # Data
    parser.add_argument('--train_csv', type=str, default='Data/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--train_dir', type=str, default='Data/train',
                        help='Path to training videos directory')
    
    # Model
    parser.add_argument('--variant', type=str, default='r3d', choices=['r3d', 'mc3'],
                        help='I3D variant (r3d or mc3)')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Video processing
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to sample from each video')
    parser.add_argument('--frame_size', type=int, default=224,
                        help='Frame size (height and width)')
    parser.add_argument('--sampling', type=str, default='uniform', 
                        choices=['uniform', 'random'],
                        help='Frame sampling strategy')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (smaller for full videos)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    
    # Save
    parser.add_argument('--save_dir', type=str, default='checkpoints/i3d',
                        help='Directory to save checkpoints')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("🚀 ENTRAÎNEMENT I3D (End-to-End)")
    print("="*70)
    
    print(f"\n📊 Configuration:")
    print(f"   • Variant: {args.variant.upper()}")
    print(f"   • Pretrained: {args.pretrained}")
    print(f"   • Num frames: {args.num_frames}")
    print(f"   • Frame size: {args.frame_size}x{args.frame_size}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Learning rate: {args.lr}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Transforms
    print(f"\n📹 Création des transforms...")
    train_transform = get_video_transforms(
        mode='train',
        img_size=args.frame_size,
        num_frames=args.num_frames
    )
    
    val_transform = get_video_transforms(
        mode='val',
        img_size=args.frame_size,
        num_frames=args.num_frames
    )
    
    # Datasets
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
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    print(f"\n✅ Datasets créés:")
    print(f"   • Train: {len(train_dataset)} vidéos")
    print(f"   • Val: {len(val_dataset)} vidéos")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n✅ DataLoaders créés:")
    print(f"   • Train batches: {len(train_loader)}")
    print(f"   • Val batches: {len(val_loader)}")
    
    # Model
    print(f"\n🏗️  Création du modèle I3D ({args.variant.upper()})...")
    model = I3D(
        num_classes=2,
        pretrained=args.pretrained,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Modèle créé:")
    print(f"   • Total paramètres: {total_params:,}")
    print(f"   • Paramètres entraînables: {trainable_params:,}")
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # Scheduler (not used - Trainer handles it internally)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.5,
    #     patience=3
    # )
    
    # Trainer
    print(f"\n⚙️  Création du Trainer...")
    
    # Create config dictionary (matches Trainer API signature)
    config = {
        'num_epochs': args.epochs,
        'patience': 10,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'save_best_only': True
    }
    
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
    print(f"   • Device: {device}")
    
    # Train
    print(f"\n🎯 Début de l'entraînement...")
    print("="*70)
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    
    print(f"\n📊 Résultats finaux:")
    print(f"   • Meilleure accuracy: {trainer.best_val_acc:.4f}")
    
    # Check if best metrics are available
    if hasattr(trainer, 'best_val_loss'):
        print(f"   • Meilleure loss: {trainer.best_val_loss:.4f}")
    if hasattr(trainer, 'best_metrics') and 'best_val_ap' in trainer.best_metrics:
        print(f"   • Meilleur AP: {trainer.best_metrics['best_val_ap']:.4f}")
    
    print(f"\n📁 Résultats sauvegardés dans: {args.save_dir}")
    print(f"   • Meilleur modèle: {args.save_dir}/best_model.pth")
    print(f"   • Métriques: {args.save_dir}/metrics.json")
    print(f"   • Log: {args.save_dir}/training.log")


if __name__ == '__main__':
    main()
