"""
Trainer pour Classification Vidéo
==================================

Gère l'entraînement, validation, métriques et checkpointing
pour tous les modèles de classification vidéo.

Auteur: Jerome
Date: Octobre 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score


class Trainer:
    """
    Classe pour entraîner les modèles de classification vidéo.
    
    Args:
        model: Modèle PyTorch
        train_loader: DataLoader pour train
        val_loader: DataLoader pour validation
        criterion: Loss function
        optimizer: Optimizer PyTorch
        device: Device (cuda/mps/cpu)
        config: Dict avec configuration
        save_dir: Dossier pour sauvegarder
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Scheduler (optionnel)
        self.scheduler = self._create_scheduler()
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aps = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience = config.get('patience', 10)
        
        # Mixed Precision (seulement pour CUDA)
        self.use_amp = device.type == 'cuda' and config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.log_file = self.save_dir / 'training.log'
        self._init_log()
        
        print(f"✅ Trainer initialisé:")
        print(f"   • Device: {device}")
        print(f"   • Mixed Precision: {'ON' if self.use_amp else 'OFF'}")
        print(f"   • Train batches: {len(train_loader)}")
        print(f"   • Val batches: {len(val_loader)}")
        print(f"   • Save dir: {save_dir}")
    
    def _create_scheduler(self):
        """Crée le learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 50),
                eta_min=1e-6
            )
        else:
            return None
    
    def _init_log(self):
        """Initialise le fichier de log."""
        with open(self.log_file, 'w') as f:
            f.write(f"Training started: {datetime.now()}\n")
            f.write(f"Config: {json.dumps(self.config, indent=2)}\n")
            f.write("="*70 + "\n")
    
    def _log(self, message: str):
        """Écrit dans le log."""
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        print(message)
    
    def train_epoch(self, epoch: int) -> float:
        """
        Entraîne le modèle pour une epoch.
        
        Returns:
            float: Loss moyenne
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision Training
            if self.use_amp:
                with autocast():
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training (CPU/MPS)
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Métriques
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """
        Valide le modèle.
        
        Returns:
            Tuple[float, Dict]: Loss et métriques
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for frames, labels in pbar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Métriques
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob classe positive
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculer métriques
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_metrics(all_labels, all_predictions, all_probs)
        
        return avg_loss, metrics
    
    def _compute_metrics(self, labels, predictions, probs) -> Dict:
        """Calcule les métriques de classification."""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        ap = average_precision_score(labels, probs)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aps': self.val_aps,
            'config': self.config
        }
        
        # Checkpoint régulier
        path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, path)
        
        # Best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self._log(f"💾 Best model saved at epoch {epoch+1}")
    
    def load_checkpoint(self, path: str):
        """Charge un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_aps = checkpoint['val_aps']
        self._log(f"✅ Checkpoint loaded: {path}")
        return checkpoint['epoch']
    
    def train(self, num_epochs: int):
        """
        Boucle d'entraînement complète.
        
        Args:
            num_epochs: Nombre d'epochs
        """
        self._log(f"\n{'='*70}")
        self._log(f"🚀 DÉBUT DE L'ENTRAÎNEMENT - {num_epochs} epochs")
        self._log(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(metrics['accuracy'])
            self.val_aps.append(metrics['ap'])
            
            # Log
            self._log(f"\nEpoch {epoch+1}/{num_epochs}:")
            self._log(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            self._log(f"  Val   - Loss: {val_loss:.4f}, Acc: {metrics['accuracy']:.4f}")
            self._log(f"  Val   - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            self._log(f"  Val   - F1: {metrics['f1']:.4f}, AP: {metrics['ap']:.4f}")
            
            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self._log(f"  Learning rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = metrics['accuracy']
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self._log(f"\n⚠️  Early stopping à epoch {epoch+1}")
                self._log(f"   Pas d'amélioration depuis {self.patience} epochs")
                break
        
        # Résumé final
        self._log(f"\n{'='*70}")
        self._log(f"✅ ENTRAÎNEMENT TERMINÉ")
        self._log(f"{'='*70}")
        self._log(f"Meilleure accuracy: {self.best_val_acc:.4f}")
        self._log(f"Meilleure loss: {self.best_val_loss:.4f}")
        self._log(f"Meilleur AP: {max(self.val_aps):.4f}")
        
        # Sauvegarder métriques
        self._save_metrics()
    
    def _save_metrics(self):
        """Sauvegarde les métriques dans un fichier JSON."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aps': self.val_aps,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        metrics_file = self.save_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self._log(f"📊 Métriques sauvegardées: {metrics_file}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict
) -> Trainer:
    """
    Helper pour créer un Trainer.
    
    Args:
        model: Modèle PyTorch
        train_loader: DataLoader train
        val_loader: DataLoader val
        config: Configuration dict
        
    Returns:
        Trainer configuré
    """
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_type = config.get('optimizer', 'adam')
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Optimizer inconnu: {optimizer_type}")
    
    # Créer trainer
    save_dir = config.get('save_dir', 'checkpoints')
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        save_dir=save_dir
    )
    
    return trainer


def test_trainer():
    """Test du Trainer avec données factices."""
    print("🧪 TEST DU TRAINER\n")
    print("="*70)
    
    # Créer modèle factice
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(16, 2)
        
        def forward(self, x):
            # x: (B, T, C, H, W) → (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Créer DataLoaders factices
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            frames = torch.randn(8, 3, 64, 64)  # 8 frames, 64x64
            label = torch.randint(0, 2, (1,)).item()
            return frames, label
    
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(20)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Config
    config = {
        'optimizer': 'adam',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'reduce_on_plateau',
        'patience': 3,
        'epochs': 5,
        'save_dir': 'test_checkpoints'
    }
    
    # Créer modèle et trainer
    model = DummyModel()
    trainer = create_trainer(model, train_loader, val_loader, config)
    
    # Entraîner
    print("\n🚀 Début de l'entraînement de test...\n")
    trainer.train(num_epochs=5)
    
    print("\n" + "="*70)
    print("✅ TEST RÉUSSI!")
    print("="*70)


if __name__ == "__main__":
    test_trainer()
