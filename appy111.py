import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle de prédiction
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # Assurez-vous que le fichier existe

model = load_model()

# Fonction d'inférence
def inference(input_data):
    return model.predict(input_data)[0]

# Initialisation de l'état de session
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Interface utilisateur
st.title("🏡 Prédiction du Prix des Maisons")
st.write("Ajustez les caractéristiques et prédisez le prix de la maison.")

# Liste des caractéristiques
test_data = pd.DataFrame(np.random.randint(0, 3000, size=(1, 10)), columns=[f"Feature_{i}" for i in range(10)])
features = test_data.columns.tolist()

# Interface utilisateur - sliders
for feature in features:
    if feature not in st.session_state.input_data:
        st.session_state.input_data[feature] = 100  # Valeur par défaut
    st.session_state.input_data[feature] = st.slider(feature, 0, 3000, st.session_state.input_data[feature])

# Convertir en DataFrame
input_df = pd.DataFrame([st.session_state.input_data])

# Bouton de prédiction
if st.button("🔍 Prédire le Prix de la Maison"):
    try:
        st.session_state.prediction = inference(input_df.values)
        st.session_state.prediction_history.append(st.session_state.prediction)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")

# Affichage du résultat
if st.session_state.prediction is not None:
    st.subheader("📢 Prédiction du Prix :")
    st.markdown(f"<h3 style='color:green;'>💰 {st.session_state.prediction:.2f} dollars</h3>", unsafe_allow_html=True)

# Historique des prédictions
if st.session_state.prediction_history:
    st.subheader("📜 Historique des Prédictions")
    st.write(pd.DataFrame(st.session_state.prediction_history, columns=["Prix Prédit"]))
