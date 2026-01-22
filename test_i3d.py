#!/usr/bin/env python3
"""
Script de test I3D pour génération de submission Kaggle
Utilise le meilleur checkpoint (epoch 13, AP=77.53%)
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    BASE_DIR = "/content/drive/MyDrive/Nexar-Dashcam-Crash-Prediction-Challenge"
    TEST_CSV = f"{BASE_DIR}/Data/test.csv"
    TEST_DIR = f"{BASE_DIR}/Data/test"
    CHECKPOINT = f"{BASE_DIR}/checkpoints/i3d/checkpoint_epoch_13.pth"
    OUTPUT_CSV = f"{BASE_DIR}/submission_i3d.csv"
    
    NUM_FRAMES = 8
    FRAME_SIZE = 160
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# VIDEO DATASET POUR TEST
# ============================================================================
class TestVideoDataset(Dataset):
    def __init__(self, csv_path, video_dir, num_frames=8, frame_size=160):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Loaded {len(self.df)} test videos")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        video_id = str(self.df.iloc[idx]['id']).zfill(5)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        frames = self._extract_frames(video_path)
        
        transformed_frames = []
        for frame in frames:
            frame_tensor = self.transform(frame)
            transformed_frames.append(frame_tensor)
        
        video_tensor = torch.stack(transformed_frames, dim=0)
        return video_tensor, video_id
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return [np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8) 
                    for _ in range(self.num_frames)]
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        
        cap.release()
        return frames


# ============================================================================
# MODÈLE I3D (ARCHITECTURE IDENTIQUE À L'ENTRAÎNEMENT)
# ============================================================================
class I3D(nn.Module):
    """I3D model - Architecture identique au checkpoint"""
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(I3D, self).__init__()
        
        from torchvision.models.video import r3d_18, R3D_18_Weights
        
        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)
        
        in_features = self.backbone.fc.in_features
        
        # Architecture identique au checkpoint
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout = dropout
        
    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        logits = self.backbone(x)
        return logits


# ============================================================================
# FONCTION DE TEST
# ============================================================================
def test_model(model, dataloader, device):
    model.eval()
    predictions = []
    video_ids = []
    
    print("\n🔮 Génération des prédictions...")
    
    with torch.no_grad():
        for videos, ids in tqdm(dataloader, desc="Testing"):
            videos = videos.to(device)
            
            # Prédiction (2 classes)
            outputs = model(videos)
            
            # Softmax pour obtenir probabilités
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Proba classe positive
            probs = probs.cpu().numpy()
            
            predictions.extend(probs)
            video_ids.extend(ids)
    
    return video_ids, predictions


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def main():
    print("=" * 70)
    print("🚀 TEST I3D - GÉNÉRATION SUBMISSION KAGGLE")
    print("=" * 70)
    
    config = Config()
    
    print(f"\n📊 Configuration:")
    print(f"   • Device: {config.DEVICE}")
    print(f"   • Checkpoint: {config.CHECKPOINT}")
    print(f"   • Test CSV: {config.TEST_CSV}")
    print(f"   • Output: {config.OUTPUT_CSV}")
    print(f"   • Num frames: {config.NUM_FRAMES}")
    print(f"   • Frame size: {config.FRAME_SIZE}x{config.FRAME_SIZE}")
    
    # Vérifications
    assert os.path.exists(config.TEST_CSV), f"❌ Test CSV not found"
    assert os.path.exists(config.TEST_DIR), f"❌ Test directory not found"
    assert os.path.exists(config.CHECKPOINT), f"❌ Checkpoint not found"
    
    # Dataset
    print("\n📹 Création du dataset...")
    test_dataset = TestVideoDataset(
        csv_path=config.TEST_CSV,
        video_dir=config.TEST_DIR,
        num_frames=config.NUM_FRAMES,
        frame_size=config.FRAME_SIZE
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✅ DataLoader créé: {len(test_loader)} batches")
    
    # Charger le modèle
    print("\n🏗️  Chargement du modèle...")
    model = I3D(num_classes=2, pretrained=False, dropout=0.5)
    
    checkpoint = torch.load(config.CHECKPOINT, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    
    epoch = checkpoint.get('epoch', 'N/A')
    best_ap = checkpoint.get('best_ap', 'N/A')
    best_acc = checkpoint.get('best_acc', 'N/A')
    
    print(f"✅ Modèle chargé (Epoch {epoch})")
    if isinstance(best_ap, (int, float)):
        print(f"   • Best AP: {best_ap:.4f}")
        print(f"   • Best Acc: {best_acc:.4f}")
    else:
        print(f"   • Best AP: {best_ap}")
        print(f"   • Best Acc: {best_acc}")
    
    # Test
    video_ids, predictions = test_model(model, test_loader, config.DEVICE)
    
    # Créer submission
    print("\n💾 Création du fichier de submission...")
    submission_df = pd.DataFrame({
        'id': video_ids,
        'target': predictions
    })
    
    # Vérifications
    print(f"\n🔍 Vérifications:")
    print(f"   • Nombre de prédictions: {len(submission_df)}")
    print(f"   • Valeurs min/max: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
    print(f"   • Moyenne: {np.mean(predictions):.4f}")
    print(f"   • Prédictions hors [0,1]: {sum((p < 0) | (p > 1) for p in predictions)}")
    
    # Sauvegarder
    submission_df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"\n✅ Submission sauvegardée: {config.OUTPUT_CSV}")
    
    # Aperçu
    print(f"\n📋 Aperçu de la submission:")
    print(submission_df.head(10))
    
    # Distribution
    print(f"\n📊 Distribution des prédictions:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(predictions, bins=bins)
    for i in range(len(bins)-1):
        print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} vidéos ({hist[i]/len(predictions)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ TEST TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print(f"\n📤 Soumettez: {config.OUTPUT_CSV}")
    print("   sur https://www.kaggle.com/competitions/nexar-collision-prediction/submit")


if __name__ == "__main__":
    main()
