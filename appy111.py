#Appy11.py     ok ce 02 avril mais à améliorer pour deploiment

# Importation des bibliothèques nécessaires
import os # os : Permet d'interagir avec le système d'exploitation. Utilisé pour manipuler les chemins de fichiers.
import pandas as pd # pandas : Bibliothèque pour la manipulation et l'analyse de données. Elle permet de travailler facilement avec des DataFrames.
import streamlit as st # streamlit : Framework permettant de créer des applications web interactives. Il est utilisé ici pour l'interface utilisateur
import numpy as np # numpy : Bibliothèque pour la manipulation de tableaux multidimensionnels et de calculs numériques. Elle est utilisée pour préparer les données avant la prédiction.
import joblib # joblib : Permet de sérialiser et de désérialiser des objets Python. Ici, on l'utilise pour charger le modèle pré-entraîné de Machine Learning.
import altair as alt # altair : Bibliothèque de visualisation de données déclarative. Elle permet de créer des graphiques interactifs pour mieux comprendre les données

# --- TITRE ET DESCRIPTION ---
st.markdown("<h1 style='color:blue; font-weight:bold;'>Prédiction du Prix d'une Maison</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:green;'>Application réalisée par <b>Fidèle Ledoux</b></h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>Cette application utilise un modèle de Machine Learning pour prédire le prix d'une maison en fonction de ses caractéristiques.</p>", unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ---
model = joblib.load("best_regression_model.joblib")

# --- FONCTION DE PRÉDICTION ---
def inference(data):
    pred = model.predict(data.reshape(1, -1))
    return pred[0]

# --- SIDEBAR : INPUTS UTILISATEUR ---
st.sidebar.header("Entrez les Caractéristiques de la Maison")

# Inputs numériques
features = [
    "Qualité globale (OverallQual)",
    "Année de construction (YearBuilt)",
    "Année de rénovation (YearRemodAdd)",
    "Surface totale du sous-sol (TotalBsmtSF)",
    "Surface du 1er étage (1stFlrSF)",
    "Surface habitable (GrLivArea)",
    "Nombre de salles de bain complètes (FullBath)",
    "Nombre total de pièces (TotRmsAbvGrd)",
    "Nombre de places de garage (GarageCars)",
    "Surface du garage (GarageArea)"
]

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(feature, min_value=0, value=100)

# --- RÉSUMÉ DES DONNÉES SAISIES ---
input_df = pd.DataFrame(input_data, index=[0])
st.write(input_df)

# --- PREDICTION ---
prediction = None
if st.button("Prédire le Prix de la Maison"):
    try:
        prediction = inference(input_df.values)
        st.subheader("Prédiction du Prix :")
        st.markdown(f"<h3 style='color:green;'>Le prix estimé de la maison est : <b>{prediction:.2f} dollars</b></h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")

# --- VISUALISATION INTERACTIVE ---
st.subheader("Visualisation Interactive des Caractéristiques")
df_melt = input_df.melt(var_name="Caractéristique", value_name="Valeur")

chart = alt.Chart(df_melt).mark_bar().encode(
    x=alt.X("Caractéristique:N", sort=None, title="Caractéristiques"),
    y=alt.Y("Valeur:Q", title="Valeur"),
    color=alt.Color("Caractéristique:N", legend=None),
    tooltip=["Caractéristique", "Valeur"]
).properties(
    width=800,
    height=400
)

st.altair_chart(chart, use_container_width=True)

# --- MESSAGE DE RETOUR ET DESIGN AMÉLIORÉ ---
st.markdown("<p style='font-size:16px;'>N'oubliez pas que cette prédiction repose sur un modèle de Machine Learning entraîné avec des données historiques. Les résultats sont des estimations et peuvent varier en fonction de plusieurs facteurs.</p>", unsafe_allow_html=True)

# --- ANALYSE DE SENSIBILITÉ ---
st.subheader("Analyse de Sensibilité")
st.markdown("Modifiez les caractéristiques pour voir comment elles influencent le prix prédit.")

# Permettre à l'utilisateur de modifier les caractéristiques et voir l'impact en temps réel
for feature in features:
    input_data[feature] = st.slider(feature, 0, 3000, input_data[feature])

updated_input_df = pd.DataFrame(input_data, index=[0])
st.write(updated_input_df)

if st.button("Mettre à jour la Prédiction"):
    try:
        updated_prediction = inference(updated_input_df.values)
        st.subheader("Prédiction Mise à Jour :")
        st.markdown(f"<h3 style='color:green;'>Le prix estimé de la maison est : <b>{updated_prediction:.2f} dollars</b></h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")

# --- HISTORIQUE DES PRÉDICTIONS ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if st.button("Ajouter à l'Historique") and prediction is not None:
    st.session_state.prediction_history.append({
        "Caractéristiques": input_data.copy(),
        "Prix Prédit": prediction
    })

st.subheader("Historique des Prédictions")
for entry in st.session_state.prediction_history:
    st.write(f"Caractéristiques: {entry['Caractéristiques']}")
    st.write(f"Prix Prédit: {entry['Prix Prédit']:.2f} dollars")

# --- CHATBOT ---
st.sidebar.subheader("Chatbot")
user_input = st.sidebar.text_input("Posez une question :")

if user_input.lower() in ["hello", "hi", "bonjour"]:
    st.sidebar.write("Bonjour ! Comment puis-je vous aider ?")
elif user_input.lower() in ["comment ça marche ?", "comment fonctionne l'application ?"]:
    st.sidebar.write("Cette application utilise un modèle de Machine Learning pour prédire le prix d'une maison en fonction de ses caractéristiques.")
else:
    st.sidebar.write("Désolé, je ne comprends pas votre question.")

# --- CHOIX DE DEVISES ---
st.sidebar.subheader("Choisir la Devise")
currency = st.sidebar.selectbox("Sélectionnez la devise :", ["USD", "EUR", "GBP"])

# Convertir le prix en fonction de la devise choisie
if currency == "EUR":
    conversion_rate = 0.85  # Exemple de taux de conversion
elif currency == "GBP":
    conversion_rate = 0.75  # Exemple de taux de conversion
else:
    conversion_rate = 1.0

if prediction is not None:
    converted_prediction = prediction * conversion_rate
    st.markdown(f"<h3 style='color:green;'>Le prix estimé de la maison est : <b>{converted_prediction:.2f} {currency}</b></h3>", unsafe_allow_html=True)

# --- TRADUCTION ---
st.sidebar.subheader("Traduction")
language = st.sidebar.selectbox("Choisissez la langue :", ["Français", "English", "Español"])

if language == "English":
    st.markdown("<h1 style='color:blue; font-weight:bold;'>House Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:green;'>Application made by <b>Fidèle Ledoux</b></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px;'>This application uses a Machine Learning model to predict the price of a house based on its characteristics.</p>", unsafe_allow_html=True)
elif language == "Español":
    st.markdown("<h1 style='color:blue; font-weight:bold;'>Predicción del Precio de una Casa</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:green;'>Aplicación realizada por <b>Fidèle Ledoux</b></h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px;'>Esta aplicación utiliza un modelo de Machine Learning para predecir el precio de una casa en función de sus características.</p>", unsafe_allow_html=True)

# --- VISUALISATION DES IMAGES ---
st.subheader("Télécharger des Images de la Maison")
uploaded_files = st.file_uploader("Choisissez des images de la maison", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Image de la maison", use_column_width=True)



