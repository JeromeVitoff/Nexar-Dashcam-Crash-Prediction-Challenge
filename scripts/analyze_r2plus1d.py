"""
Analyse complète des résultats R(2+1)D
Génère 4 graphiques + statistiques détaillées
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_metrics(checkpoint_dir):
    """Charger les métriques depuis metrics.json"""
    metrics_path = Path(checkpoint_dir) / 'metrics.json'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_r2plus1d_analysis(metrics, save_dir='analysis/r2plus1d'):
    """Générer les 4 graphiques d'analyse"""
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_accs = [acc * 100 for acc in metrics.get('train_accuracies', metrics.get('train_accs', []))]
    val_accs = [acc * 100 for acc in metrics.get('val_accuracies', metrics.get('val_accs', []))]
    val_aps = [ap * 100 for ap in metrics.get('val_aps', [])]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Trouver les meilleurs epochs
    best_epoch = np.argmax(val_accs) + 1
    best_acc = max(val_accs)
    best_ap = max(val_aps)
    best_ap_epoch = np.argmax(val_aps) + 1
    
    # Configuration
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('R(2+1)D - Analyse Complète', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # ============================================================
    # GRAPHIQUE 1 : Evolution de la Loss
    # ============================================================
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=4)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Best (epoch {best_epoch})')
    ax1.scatter([best_epoch], [val_losses[best_epoch-1]], color='green', s=200, 
                zorder=5, edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Evolution de la Loss', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(epochs) + 1)
    
    # ============================================================
    # GRAPHIQUE 2 : Accuracy sur Validation
    # ============================================================
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, val_accs, 'g-o', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=best_acc, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Best: {best_acc:.2f}%')
    ax2.scatter([best_epoch], [best_acc], color='red', s=200, 
                zorder=5, edgecolors='black', linewidths=2)
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy sur Validation', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(epochs) + 1)
    
    # ============================================================
    # GRAPHIQUE 3 : Average Precision
    # ============================================================
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, val_aps, 'orange', marker='o', label='Val AP', linewidth=2, markersize=4)
    ax3.axhline(y=best_ap, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Best: {best_ap:.2f}%')
    ax3.scatter([best_ap_epoch], [best_ap], color='red', s=200, 
                zorder=5, edgecolors='black', linewidths=2)
    ax3.axvline(x=best_ap_epoch, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AP (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Precision', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, len(epochs) + 1)
    
    # ============================================================
    # GRAPHIQUE 4 : Ecart Train-Val Loss (Overfitting)
    # ============================================================
    ax4 = plt.subplot(2, 2, 4)
    loss_gap = [val - train for train, val in zip(train_losses, val_losses)]
    
    ax4.plot(epochs, loss_gap, color='purple', marker='o', linewidth=2, markersize=4,
             label='Gap (Val - Train)')
    ax4.fill_between(epochs, 0, loss_gap, alpha=0.3, color='purple')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Déterminer le niveau d'overfitting
    final_gap = loss_gap[-1]
    if final_gap < 0:
        status = "UNDERFITTING"
        color = "orange"
    elif final_gap < 0.05:
        status = "OPTIMAL"
        color = "green"
    elif final_gap < 0.15:
        status = "OVERFITTING LÉGER"
        color = "orange"
    else:
        status = "OVERFITTING SÉVÈRE"
        color = "red"
    
    ax4.text(0.5, 0.95, status, transform=ax4.transAxes,
             fontsize=14, fontweight='bold', color=color,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
             ha='center', va='top')
    
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Loss Gap', fontsize=12, fontweight='bold')
    ax4.set_title('Ecart Train-Val Loss (Overfitting)', fontsize=14, fontweight='bold', pad=10)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, len(epochs) + 1)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/r2plus1d_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {save_dir}/r2plus1d_analysis.png")
    plt.show()
    
    # ============================================================
    # STATISTIQUES DÉTAILLÉES
    # ============================================================
    print("\n" + "="*70)
    print(" "*12 + "STATISTIQUES DÉTAILLÉES - R(2+1)D")
    print("="*70)
    
    print(f"\n🏆 MEILLEURES PERFORMANCES:")
    print(f"   Epoch            : {best_epoch}")
    print(f"   Accuracy         : {best_acc:.2f}%")
    print(f"   Loss             : {val_losses[best_epoch-1]:.4f}")
    print(f"   AP (best)        : {best_ap:.2f}% (epoch {best_ap_epoch})")
    
    print(f"\n📈 PROGRESSION:")
    print(f"   Epoch  1  ->  Acc: {val_accs[0]:.2f}%,  AP: {val_aps[0]:.2f}%,  Loss: {val_losses[0]:.4f}")
    if len(epochs) >= 5:
        print(f"   Epoch  5  ->  Acc: {val_accs[4]:.2f}%,  AP: {val_aps[4]:.2f}%,  Loss: {val_losses[4]:.4f}")
    if len(epochs) >= 10:
        print(f"   Epoch 10  ->  Acc: {val_accs[9]:.2f}%,  AP: {val_aps[9]:.2f}%,  Loss: {val_losses[9]:.4f}")
    print(f"   Epoch {best_epoch:2d}  ->  Acc: {best_acc:.2f}%,  AP: {val_aps[best_epoch-1]:.2f}%,  Loss: {val_losses[best_epoch-1]:.4f}")
    print(f"   Epoch {len(epochs):2d}  ->  Acc: {val_accs[-1]:.2f}%,  AP: {val_aps[-1]:.2f}%,  Loss: {val_losses[-1]:.4f}")
    
    improvement = val_accs[-1] - val_accs[0]
    print(f"\n   Amélioration totale: {improvement:+.2f}%")
    
    print(f"\n⚠️  OVERFITTING ANALYSIS:")
    print(f"   Train Loss (final) : {train_losses[-1]:.4f}")
    print(f"   Val Loss (final)   : {val_losses[-1]:.4f}")
    print(f"   Gap                : {final_gap:.4f}")
    print(f"   Status             : {status}")
    
    print(f"\n⏱️  ENTRAÎNEMENT:")
    print(f"   Epochs réalisés    : {len(epochs)}/30")
    early_stopped = len(epochs) < 30
    print(f"   Early stopping     : {'Oui' if early_stopped else 'Non'}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyser les résultats R(2+1)D')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/r2plus1d',
                        help='Répertoire des checkpoints')
    parser.add_argument('--save_dir', type=str, default='analysis/r2plus1d',
                        help='Répertoire de sauvegarde des graphiques')
    
    args = parser.parse_args()
    
    print("\n🔍 Chargement des métriques R(2+1)D...")
    metrics = load_metrics(args.checkpoint_dir)
    
    print("📊 Génération des graphiques...")
    plot_r2plus1d_analysis(metrics, args.save_dir)
    
    print("✅ Analyse terminée !")
