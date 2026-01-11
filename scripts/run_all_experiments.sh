#!/bin/bash
# Script pour lancer toutes les expériences
# ==========================================

echo "🚀 LANCEMENT DE TOUTES LES EXPÉRIENCES"
echo "======================================"
echo ""

# Expérience 1: ResNet-LSTM
echo "📊 Expérience 1/7: ResNet-LSTM"
python scripts/train.py --config config/exp_001_resnet_lstm.yaml
echo ""

# Expérience 2: EfficientNet-GRU
echo "📊 Expérience 2/7: EfficientNet-GRU"
python scripts/train.py --config config/exp_002_efficientnet_gru.yaml
echo ""

# Expérience 3: I3D
echo "📊 Expérience 3/7: I3D"
python scripts/train.py --model i3d --num_frames 16 --batch_size 4 --epochs 50
echo ""

# Expérience 4: R(2+1)D
echo "📊 Expérience 4/7: R(2+1)D"
python scripts/train.py --model r2plus1d --num_frames 16 --batch_size 4 --epochs 50
echo ""

# Expérience 5: TimeSformer
echo "📊 Expérience 5/7: TimeSformer"
python scripts/train.py --config config/exp_005_timesformer.yaml
echo ""

# Expérience 6: VideoMAE
echo "📊 Expérience 6/7: VideoMAE"
python scripts/train.py --config config/exp_006_videomae.yaml
echo ""

# Expérience 7: ViViT
echo "📊 Expérience 7/7: ViViT"
python scripts/train.py --model vivit --num_frames 32 --batch_size 2 --epochs 30 --lr 5e-5
echo ""

echo "======================================"
echo "✅ TOUTES LES EXPÉRIENCES TERMINÉES!"
echo "======================================"
echo ""
echo "📁 Résultats dans: checkpoints/"
echo "📊 Pour analyser: python scripts/analyze_results.py"
