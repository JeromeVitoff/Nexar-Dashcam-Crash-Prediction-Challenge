# Mémoire Professionnel M2 MIASHS
## Analyse Comparative d'Architectures Deep Learning pour la Prédiction de Collisions Routières

**Auteur:** VITOFFODJI Adjimon  
**Encadrant:** Jérôme PASQUET  
**Université:** Paul Valéry Montpellier 3  
**Année:** 2025-2026

---

## 📁 Structure du Projet

```
memoire/
├── manuscript.tex              # Fichier principal LaTeX
├── front_page.tex              # Page de garde
├── references.bib              # Bibliographie BibTeX
├── chapters/                   # Chapitres du mémoire
│   ├── remerciements.tex
│   ├── resume.tex
│   ├── introduction.tex
│   ├── chapitre1_etat_art.tex
│   ├── chapitre2_methodologie.tex
│   ├── chapitre3_implementation.tex
│   ├── chapitre4_resultats.tex
│   ├── chapitre5_discussion.tex
│   └── conclusion.tex
├── figures/                    # Images et graphiques
│   └── (ajoutez vos figures ici)
├── Makefile                    # Pour compilation automatique
└── README.md                   # Ce fichier
```

---

## 🚀 Compilation du Mémoire

### Prérequis

Vous devez avoir une distribution LaTeX installée sur votre système :

- **Windows:** [MiKTeX](https://miktex.org/) ou [TeX Live](https://www.tug.org/texlive/)
- **macOS:** [MacTeX](https://www.tug.org/mactex/)
- **Linux:** TeX Live (via package manager)

### Packages LaTeX Requis

Le mémoire utilise les packages suivants (installés automatiquement par MiKTeX/TeX Live) :
- babel, inputenc, fontenc (support français)
- graphicx, caption, subcaption (figures)
- amsmath, amsthm, amssymb (mathématiques)
- booktabs, longtable, multirow (tableaux)
- algorithm, algpseudocode (algorithmes)
- listings (code source)
- hyperref (liens PDF)
- minitoc (table des matières par chapitre)

### Compilation Manuelle

#### Méthode 1 : Via terminal (Linux/macOS)

```bash
# Première compilation
pdflatex manuscript.tex

# Générer la bibliographie
bibtex manuscript

# Générer les minitoc
pdflatex manuscript.tex

# Finaliser (résoudre références croisées)
pdflatex manuscript.tex
```

#### Méthode 2 : Via Makefile (recommandé)

```bash
# Compiler le PDF
make

# Nettoyer les fichiers temporaires
make clean

# Nettoyer tout (y compris le PDF)
make cleanall

# Voir le PDF
make view
```

#### Méthode 3 : Via éditeur LaTeX

- **TeXstudio, TeXworks, Overleaf:** Ouvrir `manuscript.tex` et cliquer sur "Build" ou "Compile"
- Assurez-vous que l'éditeur est configuré pour utiliser pdflatex + bibtex

### Compilation sur Overleaf

1. Créer un nouveau projet sur [Overleaf](https://www.overleaf.com)
2. Uploader tous les fichiers du dossier `memoire/`
3. Définir `manuscript.tex` comme fichier principal
4. Compiler avec pdfLaTeX

---

## 📊 Ajout de Figures

### Où placer les figures

Placez tous vos graphiques, schémas et images dans le dossier `figures/` :

```
figures/
├── resnet_lstm_curves.png
├── efficientnet_gru_curves.png
├── i3d_curves.png
├── architecture_comparison.pdf
└── ...
```

### Comment insérer une figure

Dans n'importe quel chapitre, utilisez :

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{nom_fichier.png}
\caption{Description de la figure}
\label{fig:mon_label}
\end{figure}
```

Pour référencer : `\ref{fig:mon_label}` ou `Figure~\ref{fig:mon_label}`

### Formats recommandés

- **Graphiques vectoriels:** PDF, SVG (meilleur pour courbes, diagrammes)
- **Images raster:** PNG (éviter JPEG pour figures techniques)
- **Résolution minimale:** 300 DPI pour impression

---

## 📈 Graphiques à Créer

Pour compléter le mémoire, vous devriez créer les graphiques suivants :

### Chapitre 4 - Résultats

1. **Courbes d'apprentissage par modèle** (6 figures)
   - Training/Validation Loss vs. Epochs
   - Training/Validation Accuracy vs. Epochs
   - Validation AP vs. Epochs

2. **Comparaisons globales**
   - Barplot : AP par modèle
   - Barplot : Accuracy par modèle
   - Scatter plot : Accuracy vs. AP
   - Barplot : Temps d'entraînement par modèle
   - Barplot : Nombre de paramètres par modèle

3. **Courbes Precision-Recall**
   - Pour chaque modèle (optionnel)

4. **Matrices de confusion**
   - Pour les meilleurs modèles (I3D, EfficientNet-GRU, VideoMAE)

### Chapitre 5 - Discussion

1. **Analyse de l'overfitting**
   - Barplot : Gap Train-Val par modèle

2. **Impact du pré-entraînement**
   - Barplot : TimeSformer vs VideoMAE

### Suggestions Python pour générer les graphiques

```python
import matplotlib.pyplot as plt
import pandas as pd

# Exemple : Barplot AP par modèle
models = ['ResNet-LSTM', 'EfficientNet-GRU', 'I3D', 'R(2+1)D', 'TimeSformer', 'VideoMAE']
aps = [69.48, 74.95, 77.53, 76.58, 50.67, 78.84]

plt.figure(figsize=(10, 6))
plt.bar(models, aps, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.ylabel('Average Precision (%)', fontsize=12)
plt.title('Comparaison des Average Precision par Modèle', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/ap_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🔧 Personnalisation

### Modifier les informations de la page de garde

Éditez `front_page.tex` :

```latex
\textbf{Présenté par :}\\
VITOFFODJI Adjimon  % Votre nom

\textbf{Encadré par :}\\
Jérôme PASQUET  % Votre encadrant
```

### Ajouter/Supprimer des chapitres

Dans `manuscript.tex`, commenter/décommenter les lignes :

```latex
\input{chapters/chapitre_nouveau}  % Ajouter
% \input{chapters/chapitre_optionnel}  % Supprimer temporairement
```

### Modifier la bibliographie

Ajoutez vos références dans `references.bib` au format BibTeX :

```bibtex
@article{auteur2025,
  title={Titre de l'article},
  author={Auteur, Prénom},
  journal={Nom du journal},
  year={2025}
}
```

Citez dans le texte : `\cite{auteur2025}`

---

## 📋 Checklist Avant Rendu

- [ ] Toutes les figures sont présentes dans `figures/`
- [ ] Toutes les références sont citées et présentes dans `references.bib`
- [ ] La compilation ne génère aucune erreur
- [ ] La table des matières est complète
- [ ] Les listes de figures et tableaux sont correctes
- [ ] Tous les labels `\ref{}` pointent vers les bonnes sections/figures
- [ ] Relecture orthographique et grammaticale
- [ ] Vérification de la cohérence des formats (dates, unités, abréviations)
- [ ] PDF final vérifié page par page

---

## 🐛 Résolution de Problèmes

### Erreur : "File not found"

- Vérifiez que tous les chemins de fichiers sont corrects
- Sur Windows, utilisez `/` au lieu de `\` dans les chemins
- Assurez-vous que les fichiers existent dans les bons dossiers

### Bibliographie non affichée

- Exécutez la séquence complète : pdflatex → bibtex → pdflatex → pdflatex
- Vérifiez qu'il y a au moins une citation `\cite{}` dans le texte

### Figures ne s'affichent pas

- Vérifiez que le chemin dans `\includegraphics{}` est correct
- Assurez-vous que l'extension du fichier est spécifiée (.png, .pdf, etc.)
- Le package `graphicx` doit être chargé (déjà fait dans manuscript.tex)

### Erreurs de compilation LaTeX

- Lisez attentivement le message d'erreur (ligne et fichier indiqués)
- Vérifiez les accolades `{}`, crochets `[]`, et environnements `\begin{...}\end{...}`
- Commentez les sections problématiques avec `%` pour isoler l'erreur

---

## 📞 Support

Pour toute question concernant le contenu scientifique, contactez votre encadrant :
- **Jérôme PASQUET** - jerome.pasquet@univ-montp3.fr

Pour les questions techniques LaTeX, consultez :
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)
- [TeX StackExchange](https://tex.stackexchange.com/)

---

## 📄 Licence

Ce mémoire est la propriété intellectuelle de VITOFFODJI Adjimon et de l'Université Paul Valéry Montpellier 3. Tous droits réservés.

---

**Bonne rédaction ! 🎓**
