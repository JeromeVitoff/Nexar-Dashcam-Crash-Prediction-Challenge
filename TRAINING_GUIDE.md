# 🚀 Guide d'Entraînement

## 📋 Prérequis

```bash
# Vérifier que tout est en place
ls src/training/trainer.py  # ✓
ls src/models/resnet_lstm.py  # ✓
ls src/data/dataloader.py  # ✓
ls scripts/train.py  # ✓
```

## 🎯 Lancer un Entraînement

### Méthode 1: Avec fichier config (Recommandé)

```bash
python scripts/train.py --config config/exp_001_resnet_lstm.yaml
```

### Méthode 2: Avec arguments CLI

```bash
python scripts/train.py \
    --model resnet_lstm \
    --num_frames 16 \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4
```

## 📊 Les 7 Expériences

| # | Modèle | Frames | Batch | Config |
|---|--------|--------|-------|--------|
| 1 | ResNet-LSTM | 16 | 8 | exp_001_resnet_lstm.yaml |
| 2 | EfficientNet-GRU | 16 | 16 | exp_002_efficientnet_gru.yaml |
| 3 | I3D | 16 | 4 | CLI |
| 4 | R(2+1)D | 16 | 4 | CLI |
| 5 | TimeSformer | 8 | 4 | exp_005_timesformer.yaml |
| 6 | VideoMAE | 16* | 4 | exp_006_videomae.yaml |
| 7 | ViViT | 32* | 2 | CLI |

*Frames fixes requis

## 🔄 Lancer Toutes les Expériences

```bash
# Rendre le script exécutable
chmod +x scripts/run_all_experiments.sh

# Lancer
./scripts/run_all_experiments.sh
```

⚠️ **Attention** : Cela peut prendre plusieurs jours !

## 📁 Structure des Résultats

```
checkpoints/
├── resnet_lstm/
│   ├── best_model.pth          # Meilleur modèle
│   ├── checkpoint_epoch_10.pth # Checkpoint epoch 10
│   ├── metrics.json            # Métriques
│   └── training.log            # Log complet
├── efficientnet_gru/
│   └── ...
└── ...
```

## 🔍 Pendant l'Entraînement

Le script affiche:
```
Epoch 1/50:
  Train - Loss: 0.6234, Acc: 65.42%
  Val   - Loss: 0.5891, Acc: 0.6823
  Val   - Precision: 0.7012, Recall: 0.6534
  Val   - F1: 0.6765, AP: 0.7234
  Learning rate: 0.000100

💾 Best model saved at epoch 1
```

## ⚠️ Contraintes Importantes

**VideoMAE** : Nécessite **EXACTEMENT 16 frames**
```bash
# ✓ Correct
python scripts/train.py --model videomae --num_frames 16

# ✗ Erreur
python scripts/train.py --model videomae --num_frames 8
```

**ViViT** : Nécessite **EXACTEMENT 32 frames**
```bash
# ✓ Correct
python scripts/train.py --model vivit --num_frames 32

# ✗ Erreur
python scripts/train.py --model vivit --num_frames 16
```

## 🛑 Arrêter l'Entraînement

`Ctrl+C` → Le checkpoint est sauvegardé automatiquement

## 📊 Analyser les Résultats

```bash
# Voir les logs
cat checkpoints/resnet_lstm/training.log

# Voir les métriques
cat checkpoints/resnet_lstm/metrics.json

# Comparer les modèles
python scripts/compare_models.py
```

## 🐛 Debugging

**Erreur de mémoire GPU** :
```bash
# Réduire batch_size
python scripts/train.py --model resnet_lstm --batch_size 4
```

**Erreur d'import** :
```bash
# Vérifier PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/train.py --model resnet_lstm
```

## 💡 Tips

1. **Commencez petit** : Testez avec 5 epochs d'abord
   ```bash
   python scripts/train.py --model resnet_lstm --epochs 5
   ```

2. **Utilisez les configs** : Plus facile à reproduire

3. **Surveillez les logs** : Regardez training.log en temps réel
   ```bash
   tail -f checkpoints/resnet_lstm/training.log
   ```

4. **Gardez le best_model.pth** : C'est lui qu'on utilise pour l'évaluation

## 📚 Prochaines Étapes

Après l'entraînement :
1. ✅ Analyser les résultats
2. ✅ Comparer les modèles
3. ✅ Évaluer sur le test set
4. ✅ Générer les tableaux pour le mémoire
