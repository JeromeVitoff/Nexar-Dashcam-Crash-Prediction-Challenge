# Makefile pour compiler le mémoire LaTeX
# Auteur: VITOFFODJI Adjimon
# Année: 2025-2026

# Variables
MAIN = manuscript
TEX = pdflatex
BIB = bibtex
VIEWER = open  # Sur macOS. Utiliser 'xdg-open' sur Linux, 'start' sur Windows

# Compiler le PDF complet
all: $(MAIN).pdf

# Compilation complète avec bibliographie
$(MAIN).pdf: $(MAIN).tex chapters/*.tex references.bib
	@echo "🔨 Compilation LaTeX (pass 1/4)..."
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	@echo "📚 Génération de la bibliographie..."
	$(BIB) $(MAIN)
	@echo "🔨 Compilation LaTeX (pass 2/4)..."
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	@echo "🔨 Compilation LaTeX (pass 3/4)..."
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	@echo "🔨 Compilation LaTeX (pass 4/4)..."
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	@echo "✅ Compilation terminée: $(MAIN).pdf"

# Compilation rapide (sans bibliographie)
quick:
	@echo "⚡ Compilation rapide..."
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	@echo "✅ Compilation rapide terminée"

# Voir le PDF
view: $(MAIN).pdf
	@echo "👀 Ouverture du PDF..."
	$(VIEWER) $(MAIN).pdf

# Nettoyer les fichiers auxiliaires
clean:
	@echo "🧹 Nettoyage des fichiers temporaires..."
	rm -f $(MAIN).aux
	rm -f $(MAIN).log
	rm -f $(MAIN).toc
	rm -f $(MAIN).lof
	rm -f $(MAIN).lot
	rm -f $(MAIN).out
	rm -f $(MAIN).bbl
	rm -f $(MAIN).blg
	rm -f $(MAIN).mtc*
	rm -f $(MAIN).maf
	rm -f chapters/*.aux
	@echo "✅ Nettoyage terminé"

# Nettoyer tout (y compris le PDF)
cleanall: clean
	@echo "🧹 Suppression du PDF..."
	rm -f $(MAIN).pdf
	@echo "✅ Nettoyage complet terminé"

# Compter les mots (approximatif)
wordcount:
	@echo "📊 Comptage des mots..."
	@texcount -inc -sum $(MAIN).tex

# Vérifier les erreurs LaTeX
check:
	@echo "🔍 Vérification des erreurs..."
	$(TEX) -interaction=nonstopmode $(MAIN).tex | grep -i "error\|warning"

# Afficher l'aide
help:
	@echo "📖 Commandes disponibles:"
	@echo "  make          - Compiler le PDF complet (avec bibliographie)"
	@echo "  make quick    - Compilation rapide (sans bibliographie)"
	@echo "  make view     - Ouvrir le PDF"
	@echo "  make clean    - Nettoyer les fichiers temporaires"
	@echo "  make cleanall - Nettoyer tout (y compris le PDF)"
	@echo "  make wordcount - Compter les mots"
	@echo "  make check    - Vérifier les erreurs"
	@echo "  make help     - Afficher cette aide"

# Déclarer les cibles qui ne sont pas des fichiers
.PHONY: all quick view clean cleanall wordcount check help
