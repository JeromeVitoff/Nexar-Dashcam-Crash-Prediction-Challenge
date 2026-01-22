#!/usr/bin/env python3
"""
Script de génération des graphiques pour le mémoire
Auteur: VITOFFODVI Adjimon
Date: Janvier 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration matplotlib pour publication
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Créer le dossier figures s'il n'existe pas
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Données des modèles
MODELS_DATA = {
    'ResNet-LSTM': {
        'accuracy': 67.33,
        'ap': 69.48,
        'train_acc': 82.50,
        'params': 2.9,  # millions
        'time': 3.0,  # heures
        'pretrain': 'ImageNet',
        'family': 'CNN-RNN'
    },
    'EfficientNet-GRU': {
        'accuracy': 71.00,
        'ap': 74.95,
        'train_acc': 88.25,
        'params': 3.6,
        'time': 3.0,
        'pretrain': 'ImageNet',
        'family': 'CNN-RNN'
    },
    'I3D': {
        'accuracy': 70.00,
        'ap': 77.53,
        'train_acc': 99.58,
        'params': 33.3,
        'time': 6.0,
        'pretrain': 'Kinetics',
        'family': '3D CNN'
    },
    'R(2+1)D': {
        'accuracy': 68.67,
        'ap': 76.58,
        'train_acc': 96.83,
        'params': 32.0,
        'time': 5.0,
        'pretrain': 'Kinetics',
        'family': '3D CNN'
    },
    'TimeSformer': {
        'accuracy': 50.67,
        'ap': 50.67,
        'train_acc': 52.17,
        'params': 120.0,
        'time': 8.0,
        'pretrain': 'None',
        'family': 'Transformer'
    },
    'VideoMAE': {
        'accuracy': 68.00,
        'ap': 78.84,
        'train_acc': 99.17,
        'params': 86.0,
        'time': 7.0,
        'pretrain': 'Kinetics',
        'family': 'Transformer'
    }
}

def plot_ap_comparison():
    """Graphique 1: Comparaison des Average Precision"""
    models = list(MODELS_DATA.keys())
    aps = [MODELS_DATA[m]['ap'] for m in models]
    
    # Couleurs par famille
    colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#d62728', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, aps, color=colors, edgecolor='black', linewidth=1.2)
    
    # Ajouter les valeurs sur les barres
    for bar, ap in zip(bars, aps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{ap:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Precision (%)', fontweight='bold')
    ax.set_title('Comparaison des Average Precision par Modèle', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=77.53, color='green', linestyle='--', alpha=0.5, label='I3D (meilleur)')
    
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ap_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'ap_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique AP créé: figures/ap_comparison.pdf")
    plt.close()

def plot_accuracy_comparison():
    """Graphique 2: Comparaison des Accuracy"""
    models = list(MODELS_DATA.keys())
    accs = [MODELS_DATA[m]['accuracy'] for m in models]
    
    colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#d62728', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accs, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Comparaison des Accuracy par Modèle', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=71.00, color='orange', linestyle='--', alpha=0.5, label='EfficientNet-GRU (meilleur)')
    
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Accuracy créé: figures/accuracy_comparison.pdf")
    plt.close()

def plot_accuracy_vs_ap():
    """Graphique 3: Scatter plot Accuracy vs AP"""
    models = list(MODELS_DATA.keys())
    accs = [MODELS_DATA[m]['accuracy'] for m in models]
    aps = [MODELS_DATA[m]['ap'] for m in models]
    
    families = [MODELS_DATA[m]['family'] for m in models]
    family_colors = {'CNN-RNN': '#1f77b4', '3D CNN': '#2ca02c', 'Transformer': '#d62728'}
    colors = [family_colors[f] for f in families]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (acc, ap, model, color) in enumerate(zip(accs, aps, models, colors)):
        ax.scatter(acc, ap, s=200, c=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.annotate(model, (acc, ap), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Accuracy (%)', fontweight='bold')
    ax.set_ylabel('Average Precision (%)', fontweight='bold')
    ax.set_title('Compromis Accuracy vs Average Precision', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Légende des familles
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=family_colors[f], label=f) 
                      for f in family_colors.keys()]
    ax.legend(handles=legend_elements, title='Famille')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'accuracy_vs_ap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'accuracy_vs_ap.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Accuracy vs AP créé: figures/accuracy_vs_ap.pdf")
    plt.close()

def plot_overfitting_analysis():
    """Graphique 4: Analyse de l'overfitting (gap train-val)"""
    models = list(MODELS_DATA.keys())
    train_accs = [MODELS_DATA[m]['train_acc'] for m in models]
    val_accs = [MODELS_DATA[m]['accuracy'] for m in models]
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', 
                  color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, val_accs, width, label='Val Accuracy',
                  color='salmon', edgecolor='black')
    
    # Ajouter les gaps
    for i, gap in enumerate(gaps):
        ax.text(i, max(train_accs[i], val_accs[i]) + 2, f'Δ={gap:.1f}%',
               ha='center', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Analyse de l\'Overfitting (Gap Train-Validation)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overfitting_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Overfitting créé: figures/overfitting_analysis.pdf")
    plt.close()

def plot_training_time():
    """Graphique 5: Temps d'entraînement par modèle"""
    models = list(MODELS_DATA.keys())
    times = [MODELS_DATA[m]['time'] for m in models]
    
    colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#d62728', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, times, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
                f'{time:.1f}h',
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Temps d\'entraînement (heures)', fontweight='bold')
    ax.set_title('Temps d\'Entraînement par Modèle', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_time.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'training_time.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Temps créé: figures/training_time.pdf")
    plt.close()

def plot_parameters_count():
    """Graphique 6: Nombre de paramètres par modèle"""
    models = list(MODELS_DATA.keys())
    params = [MODELS_DATA[m]['params'] for m in models]
    
    colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#d62728', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, params, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{param:.1f}M',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel('Nombre de paramètres (millions)', fontweight='bold')
    ax.set_title('Complexité des Modèles (Nombre de Paramètres)', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'parameters_count.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'parameters_count.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Paramètres créé: figures/parameters_count.pdf")
    plt.close()

def plot_pretraining_impact():
    """Graphique 7: Impact du pré-entraînement (Transformers)"""
    models = ['TimeSformer\n(sans pré-train)', 'VideoMAE\n(avec pré-train)']
    aps = [50.67, 78.84]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, aps, color=['#d62728', '#ff7f0e'], 
                  edgecolor='black', linewidth=1.5, width=0.6)
    
    for bar, ap in zip(bars, aps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{ap:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Ajouter la différence
    ax.annotate('', xy=(0, 78.84), xytext=(0, 50.67),
               arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    ax.text(0.15, 65, f'+{78.84-50.67:.2f}%\nGain', 
           fontsize=12, fontweight='bold', color='green')
    
    ax.set_ylabel('Average Precision (%)', fontweight='bold')
    ax.set_title('Impact Critique du Pré-entraînement sur les Transformers', 
                fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pretraining_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'pretraining_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Impact Pré-training créé: figures/pretraining_impact.pdf")
    plt.close()

def plot_family_comparison():
    """Graphique 8: Comparaison par famille d'architecture"""
    families = {
        'CNN-RNN\nHybrides': [69.48, 74.95],
        '3D CNN': [77.53, 76.58],
        'Transformers\n(avec pré-train)': [78.84],
        'Transformers\n(sans pré-train)': [50.67]
    }
    
    family_names = list(families.keys())
    mean_aps = [np.mean(families[f]) for f in family_names]
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(family_names, mean_aps, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    for bar, ap in zip(bars, mean_aps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{ap:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Precision Moyenne (%)', fontweight='bold')
    ax.set_title('Performance Moyenne par Famille d\'Architecture', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'family_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'family_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique Familles créé: figures/family_comparison.pdf")
    plt.close()

def main():
    """Générer tous les graphiques"""
    print("🎨 Génération des graphiques pour le mémoire...")
    print(f"📁 Dossier de sortie: {FIGURES_DIR.absolute()}\n")
    
    plot_ap_comparison()
    plot_accuracy_comparison()
    plot_accuracy_vs_ap()
    plot_overfitting_analysis()
    plot_training_time()
    plot_parameters_count()
    plot_pretraining_impact()
    plot_family_comparison()
    
    print(f"\n✅ Tous les graphiques ont été générés dans {FIGURES_DIR}/")
    print("📄 Formats générés: PDF (vectoriel) + PNG (raster)")
    print("\n💡 Conseil: Utilisez les PDF dans votre mémoire LaTeX pour une qualité optimale.")

if __name__ == "__main__":
    main()
