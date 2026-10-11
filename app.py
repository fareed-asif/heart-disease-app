"""
Purpose of this file:
----------------------
This is the Streamlit web app for the Heart Disease Prediction System.
It loads the trained model, takes four inputs from the user through a web
form, converts them into numbers using Label Encoders (exactly like the
notebook), and displays the prediction.
"""

# Import required libraries
import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Find the folder of this file, so the model is found on the cloud as well
BASE = os.path.dirname(os.path.abspath(__file__))

# Load the saved Support Vector Classifier model
model = pickle.load(open(os.path.join(BASE, "heart_svc_trained_model.pkl"), "rb"))

# Possible values of each input attribute (same as the notebook)
chestpain_options = ["Typical", "Atypical", "NonAnginal", "Asymptomatic"]
gender_options = ["Male", "Female"]
vessel_options = ["Zero", "One", "Two", "Three"]
angina_options = ["Yes", "No"]

# Create and train the Label Encoders (same categories as the notebook)
def make_encoder(values):
    encoder = LabelEncoder()
    encoder.fit(values)
    return encoder

chestpain_enc = make_encoder(chestpain_options)
gender_enc = make_encoder(gender_options)
vessel_enc = make_encoder(vessel_options)
angina_enc = make_encoder(angina_options)

# Page title and description
st.title("Heart Disease Prediction System")
st.write("Enter four patient attributes. For learning only, not medical advice.")

# Take input from the user
chestpain = st.selectbox("Chest pain type", chestpain_options)
gender = st.selectbox("Gender", gender_options)
vessels = st.selectbox("Number of major vessels (0-3)", vessel_options)
angina = st.selectbox("Exercise induced angina", angina_options)

# Predict when the button is clicked
if st.button("Predict"):
    # Convert user input into a label encoded feature vector (same column order as training)
    features = pd.DataFrame({
        "ChestPain": chestpain_enc.transform([chestpain]),
        "Gender": gender_enc.transform([gender]),
        "MajorVessels": vessel_enc.transform([vessels]),
        "ExerciseAngina": angina_enc.transform([angina]),
    })

    # Apply the model and show the prediction
    result = model.predict(features)[0]
    if result == 1:
        st.error("Prediction: HEART DISEASE")
    else:
        st.success("Prediction: NO HEART DISEASE")
