# Nexar-Dashcam-Crash-Prediction-Challenge
# Projet Semestre 1 Master 2 MIASHS 

Ce Challenge a été realisé dans le cadre du projet du premier semestre. 

# Structure de Projet

nexar-collision-prediction/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── config.yaml          # Hyperparamètres centralisés
│   └── experiments/         # Configs par expérience
├── data/
│   ├── raw/                 # train.csv, test.csv
│   └── processed/           # Données prétraitées
├── notebooks/
│   ├── 01_EDA.ipynb        # Analyse exploratoire
│   ├── 02_baseline.ipynb   # Modèle de base
│   └── 03_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py      # PyTorch Dataset
│   │   ├── transforms.py   # Augmentations
│   │   └── loader.py       # DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet_lstm.py
│   │   ├── i3d.py
│   │   ├── timesformer.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/
│       ├── visualization.py
│       └── helpers.py
├── experiments/            # Résultats des expériences
│   └── exp_001/
│       ├── config.yaml
│       ├── logs/
│       ├── checkpoints/
│       └── results.json
└── scripts/
    ├── train.py           # Script d'entraînement
    ├── evaluate.py
    └── submit.py



    
