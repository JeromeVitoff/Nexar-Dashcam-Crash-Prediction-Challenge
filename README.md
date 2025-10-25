# Nexar-Dashcam-Crash-Prediction-Challenge
# Projet Semestre 1 Master 2 MIASHS 

# À Propos:
Ce projet de recherche s'inscrit dans le cadre du Mémoire de Master 2 et vise à développer des modèles de deep learning capables de prédire les collisions automobiles avant qu'elles ne se produisent, en analysant des vidéos de dashcam en temps réel.

# Objectif Principal : Développer un système d'alerte précoce pour les véhicules autonomes et les systèmes ADAS (Advanced Driver Assistance Systems) afin d'améliorer la sécurité routière.

# Challenge Kaggle
- Competition : Nexar Collision Prediction Challenge
- Métrique d'évaluation : Mean Average Precision (mAP) à 500ms, 1000ms et 1500ms
- Dataset : 1,500 vidéos d'entraînement + 1,344 vidéos de test (31 GB)


# Lien du projet :

https://www.kaggle.com/competitions/nexar-collision-prediction/overview


# Structure de Projet

nexar-collision-prediction/

├── config/              # Configurations YAML
├── data/               # Données (gitignored)
├── notebooks/          # Jupyter notebooks
├── src/                # Code source
│   ├── data/          # Dataset & DataLoader
│   ├── models/        # Architectures
│   ├── training/      # Trainer & Metrics
│   └── utils/         # Utilitaires
├── experiments/        # Résultats des expériences
├── scripts/           # Scripts exécutables
├── results/           # Résultats agrégés
└── docs/              # Documentation



##  Structure  Complète

Nexar-Dashcam-Crash-Prediction-Challenge/
│
├── requirements.txt          ← À la racine
├── .gitignore               ← À la racine
├── README.md                ← À la racine
├── setup.py                 ← À la racine (optionnel pour l'instant)
│
├── Data/                    ← Vos données (déjà existant)
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train/              # 1,500 vidéos
│   └── test/               # 1,344 vidéos
│
├── notebooks/              ← À créer
│   └── 01_EDA.ipynb       ← Le notebook qu'on va créer
│
├── src/                    ← À créer plus tard
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   └── utils/
│
├── experiments/            ← À créer plus tard
├── results/               ← À créer plus tard
└── scripts/               ← À créer plus tard
    
