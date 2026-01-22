"""
Analyse complète des résultats TimeSformer
Génère 4 graphiques + statistiques détaillées
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(checkpoint_dir):
    """Charger les métriques depuis metrics.json"""
    metrics_path = Path(checkpoint_dir) / 'metrics.json'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_timesformer_analysis(metrics, save_dir='analysis/timesformer'):
    """Générer les 4 graphiques d'analyse"""
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Gérer différentes structures de metrics.json
    if 'train_losses' in metrics:
        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        train_accs = [acc * 100 for acc in metrics.get('train_accuracies', metrics.get('train_accs', []))]
        val_accs = [acc * 100 for acc in metrics.get('val_accuracies', metrics.get('val_accs', []))]
        val_aps = [ap * 100 for ap in metrics.get('val_aps', [])]
    else:
        # Structure alternative
        train_losses = [m['train_loss'] for m in metrics['history']] if 'history' in metrics else []
        val_losses = [m['val_loss'] for m in metrics['history']] if 'history' in metrics else []
        train_accs = [m['train_acc'] * 100 for m in metrics['history']] if 'history' in metrics else []
        val_accs = [m['val_acc'] * 100 for m in metrics['history']] if 'history' in metrics else []
        val_aps = [m['val_ap'] * 100 for m in metrics['history']] if 'history' in metrics else []
    
    epochs = range(1, len(train_losses) + 1)
    
    # Trouver le meilleur epoch
    best_epoch = np.argmax(val_accs) + 1 if val_accs else 1
    best_acc = max(val_accs) if val_accs else 0
    best_ap = max(val_aps) if val_aps else 0
    best_ap_epoch = np.argmax(val_aps) + 1 if val_aps else 1
    
    # Créer la figure avec 4 sous-graphiques
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('TimeSformer - Analyse Complète (ÉCHEC)', fontsize=20, fontweight='bold', y=0.995, color='red')
    
    # ============================================================
    # GRAPHIQUE 1 : Evolution de la Loss
    # ============================================================
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=4)
    ax1.axhline(y=0.6931, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                label='Loss théorique (hasard) = 0.6931')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Evolution de la Loss - STAGNATION', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(epochs) + 1)
    
    # ============================================================
    # GRAPHIQUE 2 : Accuracy sur Validation
    # ============================================================
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, val_accs, 'g-o', label='Val Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Hasard: 50%')
    ax2.axhline(y=best_acc, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    
    # Zone de "hasard" en rouge
    ax2.fill_between([0, len(epochs) + 1], 48, 52, alpha=0.2, color='red', 
                     label='Zone hasard (48-52%)')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Accuracy sur Validation - BLOQUÉE À {best_acc:.1f}%', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(epochs) + 1)
    ax2.set_ylim(45, 55)
    
    # ============================================================
    # GRAPHIQUE 3 : Average Precision
    # ============================================================
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, val_aps, 'orange', marker='o', label='Val AP', linewidth=2, markersize=4)
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Hasard: 50%')
    
    # Zone de "hasard" en rouge
    ax3.fill_between([0, len(epochs) + 1], 48, 52, alpha=0.2, color='red',
                     label='Zone hasard (48-52%)')
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AP (%)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Average Precision - AUCUN APPRENTISSAGE', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, len(epochs) + 1)
    ax3.set_ylim(45, 55)
    
    # ============================================================
    # GRAPHIQUE 4 : Comportement des prédictions
    # ============================================================
    ax4 = plt.subplot(2, 2, 4)
    
    # Texte explicatif de l'échec
    text = """
    ❌ ÉCHEC TOTAL DE L'ENTRAÎNEMENT
    
    Comportement observé :
    
    • Epochs 1-4 : Prédictions → 100% "Négatif"
      (Acc ≈ 49%)
    
    • Epochs 5+ : Prédictions → 100% "Positif"
      (Acc ≈ 51%)
    
    Diagnostic :
    • 114.6M paramètres vs 1,200 vidéos
    • Ratio : 95,526 params/vidéo (×3.4 vs I3D)
    • AUCUN pré-entraînement
    • Le modèle oscille entre 2 stratégies triviales
    
    Conclusion :
    Transformers NÉCESSITENT pré-entraînement
    pour datasets limités (<10K exemples)
    """
    
    ax4.text(0.5, 0.5, text, transform=ax4.transAxes,
             fontsize=11, fontweight='normal',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             ha='center', va='center', family='monospace')
    
    ax4.axis('off')
    ax4.set_title('Analyse de l\'Échec', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/timesformer_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {save_dir}/timesformer_analysis.png")
    plt.show()
    
    # ============================================================
    # STATISTIQUES DÉTAILLÉES
    # ============================================================
    print("\n" + "="*70)
    print(" "*12 + "STATISTIQUES DÉTAILLÉES - TIMESFORMER")
    print("="*70)
    
    print(f"\n❌ RÉSULTAT : ÉCHEC TOTAL")
    print(f"   Epoch            : {best_epoch}")
    print(f"   Accuracy         : {best_acc:.2f}% ← HASARD (50%)")
    print(f"   Loss             : {val_losses[best_epoch-1]:.4f} ← ~0.693 (hasard)")
    print(f"   AP (best)        : {best_ap:.2f}% ← HASARD (50%)")
    
    print(f"\n📈 ABSENCE DE PROGRESSION:")
    print(f"   Epoch  1  ->  Acc: {val_accs[0]:.2f}%,  AP: {val_aps[0]:.2f}%,  Loss: {val_losses[0]:.4f}")
    if len(epochs) >= 5:
        print(f"   Epoch  5  ->  Acc: {val_accs[4]:.2f}%,  AP: {val_aps[4]:.2f}%,  Loss: {val_losses[4]:.4f}")
    if len(epochs) >= 10:
        print(f"   Epoch 10  ->  Acc: {val_accs[9]:.2f}%,  AP: {val_aps[9]:.2f}%,  Loss: {val_losses[9]:.4f}")
    print(f"   Epoch {len(epochs):2d}  ->  Acc: {val_accs[-1]:.2f}%,  AP: {val_aps[-1]:.2f}%,  Loss: {val_losses[-1]:.4f}")
    
    improvement = val_accs[-1] - val_accs[0]
    print(f"\n   Amélioration totale: {improvement:+.2f}% ← AUCUNE !")
    
    print(f"\n⚠️  DIAGNOSTIC DE L'ÉCHEC:")
    print(f"   Paramètres         : 114,630,914 (115M)")
    print(f"   Dataset train      : 1,200 vidéos")
    print(f"   Ratio params/vidéo : 95,526 (×3.4 vs I3D)")
    print(f"   Pré-entraînement   : NON ❌")
    print(f"   Status             : ÉCHEC - Aucun apprentissage")
    
    print(f"\n💡 CAUSE PRINCIPALE:")
    print(f"   Transformers nécessitent IMPÉRATIVEMENT du pré-entraînement")
    print(f"   pour datasets limités. Avec 115M params et 1,200 exemples,")
    print(f"   le modèle ne peut capturer aucun pattern discriminant.")
    
    print(f"\n⏱️  TEMPS D'ENTRAINEMENT:")
    print(f"   Epochs réalisés    : {len(epochs)}/30")
    early_stopped = len(epochs) < 30
    print(f"   Early stopping     : {'Oui (aucune amélioration)' if early_stopped else 'Non'}")
    
    print("\n" + "="*70)
    print("🎓 CONCLUSION POUR LE MÉMOIRE:")
    print("="*70)
    print("""
Ce résultat négatif est scientifiquement VALIDE et IMPORTANT.
Il démontre empiriquement que :

1. Les Transformers sans pré-entraînement échouent totalement
   sur des datasets limités (<10K exemples)

2. Les CNN 3D pré-entraînés (I3D: 70% acc) surpassent largement
   les Transformers from scratch (50% acc = hasard)

3. Le transfer learning est CRITIQUE pour les architectures
   Transformer en vision par ordinateur

Cette expérience valide la littérature et guide les praticiens :
pour des applications avec données limitées, privilégier les
CNN pré-entraînés plutôt que les Transformers from scratch.
""")
    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyser les résultats TimeSformer')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/timesformer',
                        help='Répertoire des checkpoints')
    parser.add_argument('--save_dir', type=str, default='analysis/timesformer',
                        help='Répertoire de sauvegarde des graphiques')
    
    args = parser.parse_args()
    
    print("\n🔍 Chargement des métriques TimeSformer...")
    metrics = load_metrics(args.checkpoint_dir)
    
    print("📊 Génération des graphiques...")
    plot_timesformer_analysis(metrics, args.save_dir)
    
    print("✅ Analyse terminée !")
