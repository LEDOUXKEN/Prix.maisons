# Projet d'Analyse Prédictive Immobilière avec Scikit-Learn

Ce projet vise à analyser et prédire les prix immobiliers en utilisant des techniques d'apprentissage automatique avec Scikit-Learn. Il comprend le chargement et l'exploration des données, le nettoyage et le prétraitement, l'analyse exploratoire des données (EDA), la construction et l'évaluation de modèles, et la préparation des données pour une application Streamlit.

**Auteur:** Fidèle Ledoux

## 1. Introduction

L'objectif principal de ce projet est de développer un modèle de prédiction de prix immobiliers précis. Le projet utilise une approche de bout en bout, depuis le traitement initial des données jusqu'à la préparation pour le déploiement.

## 2. Structure du Projet

* `README.md`: Ce fichier, fournissant une vue d'ensemble du projet.
* `notebook/`: Contient le notebook Jupyter (`Real_Estate_Analysis.ipynb`) avec le code source de l'analyse et de la modélisation.
* `data/`: Inclut les jeux de données utilisés (`train.csv`) et les données préparées pour Streamlit (`streamlit_data.csv`).
* `model/`: (Optionnel) Stocke les modèles entraînés et sauvegardés (`model.pkl`).

## 3. Dépendances

Le projet utilise les bibliothèques Python suivantes :

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `missingno`
* `plotly`
* `scikit-learn`
* `xgboost`
* `joblib`
* `streamlit` (si vous utilisez cette librairie)

Pour installer les dépendances, utilisez `pip` :

```bash
pip install pandas numpy matplotlib seaborn missingno plotly scikit-learn xgboost joblib streamlit
