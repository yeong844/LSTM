# app3.py

import streamlit as st
import re
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("sentiment_lstm_model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Prediction function
def predict_sentiment(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction)
    
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    confidence = prediction[0][predicted_class] * 100
    
    return labels[predicted_class], round(confidence, 2)

# Streamlit UI
st.set_page_config(page_title="Coffee Review Sentiment", layout="centered")
st.title("☕ Coffee Review Sentiment Analyzer")

st.markdown("""
Type or paste a coffee review below and click **Predict** to see the sentiment.
""")

user_input = st.text_area("✍️ Enter a coffee review:", height=150)

if st.button("🔍 Predict Sentiment"):
    if user_input.strip():
        label, confidence = predict_sentiment(user_input)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence}%")
    else:
        st.warning("Please enter some text for prediction.")
