import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load the model
model = load_model("model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

max_len = 815 # Set the maximum length of input sequences

st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("🧠 Toxic Comment Classifier")
st.markdown("Type a comment and check if it's **TOXIC** or **NON-TOXIC**.")

# Input text
comment = st.text_area("Enter a comment:", height=150)

# Predict button
if st.button("Check Toxicity"):
    if comment.strip() == "":
        st.warning("Please enter a comment to analyze.")
    else:
        # Preprocessing
        sequence = tokenizer.texts_to_sequences([comment])
        padded = pad_sequences(sequence, maxlen=max_len)
        
        # Prediction
        prediction = model.predict(padded)[0][0]
        label = "🚫 Toxic" if prediction > 0.5 else "✅ Non-Toxic"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Output
        st.markdown(f"### Result: **{label}**")
        st.progress(float(confidence))
        st.write(f"Confidence: `{confidence:.2%}`")
