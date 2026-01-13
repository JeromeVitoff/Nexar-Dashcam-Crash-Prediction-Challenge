"""
Script de Pré-extraction de Features ResNet50
==============================================

Extrait les features ResNet50 pour toutes les vidéos UNE SEULE FOIS.
Permet ensuite d'entraîner le LSTM 10× plus vite.

Usage:
    python scripts/extract_features.py --output_dir features/resnet50

Auteur: Jerome
Date: Janvier 2026
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_transform


class FeatureExtractor(nn.Module):
    """Extracteur de features ResNet50."""
    
    def __init__(self):
        super().__init__()
        # Charger ResNet50 pré-entraîné
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Retirer la dernière couche (fc)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze les poids
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
        
        print("✅ ResNet50 Feature Extractor initialisé")
        print("   • Output dim: 2048")
        print("   • Weights: ImageNet pretrained")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            features: (batch, 2048)
        """
        features = self.features(x)  # (batch, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        return features


def load_video(video_path, num_frames=16):
    """
    Charge une vidéo et échantillonne num_frames.
    
    Returns:
        frames: (num_frames, H, W, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Échantillonnage uniforme
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    return np.array(frames)


def extract_features_from_video(video_path, extractor, transform, device, num_frames=16):
    """
    Extrait les features d'une vidéo.
    
    Returns:
        features: (num_frames, 2048) tensor
    """
    # Charger vidéo
    frames = load_video(video_path, num_frames)  # (num_frames, H, W, 3)
    
    # Transformer chaque frame
    transformed_frames = []
    for frame in frames:
        # transform attend (H, W, C)
        transformed = transform(frame)  # (3, 224, 224)
        transformed_frames.append(transformed)
    
    # Stack en batch
    batch = torch.stack(transformed_frames).to(device)  # (num_frames, 3, 224, 224)
    
    # Extraire features
    features = extractor(batch)  # (num_frames, 2048)
    
    return features.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='Data/train.csv')
    parser.add_argument('--train_dir', type=str, default='Data/train')
    parser.add_argument('--output_dir', type=str, default='features/resnet50')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_videos', type=int, default=1, help='Nombre de vidéos en parallèle')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎬 EXTRACTION DE FEATURES ResNet50")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   • CSV: {args.train_csv}")
    print(f"   • Videos dir: {args.train_dir}")
    print(f"   • Output dir: {args.output_dir}")
    print(f"   • Frames par vidéo: {args.num_frames}")
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"   • Device: {device}")
    
    # Créer output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger le CSV
    df = pd.read_csv(args.train_csv)
    print(f"\n📥 Dataset: {len(df)} vidéos")
    
    # Créer l'extracteur
    print("\n🏗️  Création du Feature Extractor...")
    extractor = FeatureExtractor().to(device)
    
    # Transform (validation mode = pas d'augmentation)
    transform = get_transform(mode='val', img_size=224, augmentation='none')
    
    # Extraction
    print(f"\n🚀 Début de l'extraction...")
    print("="*70)
    
    extracted = 0
    skipped = 0
    errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extraction"):
        video_id = row['id']
        video_path = Path(args.train_dir) / f"{video_id}.mp4"
        output_path = output_dir / f"{video_id}.pt"
        
        # Skip si déjà extrait
        if output_path.exists():
            skipped += 1
            continue
        
        # Vérifier que la vidéo existe
        if not video_path.exists():
            errors += 1
            continue
        
        try:
            # Extraire features
            features = extract_features_from_video(
                video_path, 
                extractor, 
                transform, 
                device,
                num_frames=args.num_frames
            )
            
            # Sauvegarder
            torch.save({
                'features': features,  # (num_frames, 2048)
                'video_id': video_id,
                'target': row['target'],
                'num_frames': args.num_frames
            }, output_path)
            
            extracted += 1
            
        except Exception as e:
            print(f"\n❌ Erreur pour {video_id}: {e}")
            errors += 1
            continue
    
    # Résumé
    print("\n" + "="*70)
    print("✅ EXTRACTION TERMINÉE")
    print("="*70)
    print(f"\n📊 Résumé:")
    print(f"   • Extraites: {extracted}")
    print(f"   • Skippées (déjà faites): {skipped}")
    print(f"   • Erreurs: {errors}")
    print(f"   • Total: {len(df)}")
    print(f"\n📁 Features sauvegardées dans: {output_dir}")
    print(f"   • Format: {video_id}.pt")
    print(f"   • Contenu: dict avec 'features' (num_frames, 2048)")
    
    # Vérification
    if extracted > 0:
        sample_path = list(output_dir.glob('*.pt'))[0]
        sample = torch.load(sample_path)
        print(f"\n✅ Vérification (échantillon):")
        print(f"   • Shape: {sample['features'].shape}")
        print(f"   • Device: {sample['features'].device}")
        print(f"   • Dtype: {sample['features'].dtype}")


if __name__ == "__main__":
    main()
