#   Analyse Combinée : Prédiction du Risque de Crédit et des Prix Immobiliers

Ce projet combine deux analyses distinctes : la prédiction du risque de crédit et la prédiction des prix immobiliers. Chacune utilise des techniques d'apprentissage automatique avec des bibliothèques Python telles que Scikit-Learn, Pandas, Numpy, etc.

##   1.  Prédiction du Risque de Crédit

###   1.1. Introduction

L'objectif de cette partie est de développer un modèle qui prédit avec précision le risque de crédit. L'analyse comprend le prétraitement des données, l'analyse exploratoire des données (EDA) et la construction de modèles pour évaluer la probabilité de défaut de paiement sur un prêt.

###   1.2. Description des Données

Le jeu de données (`credit_risk_dataset.csv`) contient des informations sur le risque de crédit, avec des détails sur les demandeurs de prêt et le statut de leurs prêts. Les attributs clés incluent :

* `person_age`: Âge du demandeur
* `person_income`: Revenu annuel du demandeur
* `person_home_ownership`: Statut de propriété du logement
* `person_emp_length`: Durée d'emploi en années
* `loan_amnt`: Montant du prêt
* `loan_int_rate`: Taux d'intérêt du prêt
* `loan_percent_income`: Montant du prêt en pourcentage du revenu
* `loan_status`: Statut du prêt (variable cible)
* `loan_intent`: Objectif du prêt
* `loan_grade`: Catégorie du prêt
* `cb_person_default_on_file`: Information de défaut au bureau de crédit
* `cb_person_cred_hist_length`: Durée de l'historique de crédit

###   1.3. Prétraitement des Données

Le prétraitement des données (détaillé dans `Credit_Risk_Analysis.ipynb`) comprend :

* Gestion des valeurs manquantes (imputation par la médiane ou le mode).
* Encodage des variables catégorielles (one-hot encoding ou Label Encoding).
* Mise à l'échelle des caractéristiques numériques (StandardScaler).

###   1.4. Modélisation

Plusieurs modèles d'apprentissage automatique sont utilisés, notamment la régression logistique et les arbres de décision.

###   1.5. Résultats

L'arbre de décision a montré de bonnes performances, en particulier pour minimiser les faux négatifs, ce qui est crucial dans l'évaluation du risque de crédit.

###   1.6. Déploiement

Le modèle de risque de crédit est déployé via une application Streamlit :

* [https://riskcredit-bgqhwpvq2f3mw93pcnbzum.streamlit.app/](https://riskcredit-bgqhwpvq2f3mw93pcnbzum.streamlit.app/)

##   2.  Prédiction des Prix Immobiliers

###   2.1. Introduction

Cette partie du projet vise à analyser et à prédire les prix immobiliers à l'aide de Scikit-Learn. Elle couvre le chargement et l'exploration des données, le nettoyage et le prétraitement, l'EDA, la construction et l'évaluation de modèles, et la préparation des données pour une application Streamlit.

###   2.2. Structure du Projet

* `Real_Estate_Analysis.ipynb`: Contient le notebook Jupyter avec le code source de l'analyse.
* `train.csv`: Inclut le jeu de données utilisé pour l'entraînement du modèle.
* `streamlit_data.csv`: (Optionnel) Données préparées pour l'application Streamlit.
* `model.pkl`: (Optionnel) Stocke le modèle entraîné et sauvegardé.

###   2.3. Dépendances

Bibliothèques Python utilisées (installables via `pip`) :

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `missingno`
* `plotly`
* `scikit-learn`
* `xgboost`
* `joblib`
* `streamlit`

Installation des dépendances :

```bash
pip install pandas numpy matplotlib seaborn missingno plotly scikit-learn xgboost joblib streamlit
