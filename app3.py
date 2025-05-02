import streamlit as st
import numpy as np
import joblib
import re
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# STREAMLIT PAGE SETUP
# ======================
st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================
# CONSTANTS
# ======================
MAX_LEN = 100

# ======================
# MODEL & TOKENIZER LOADING
# ======================
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model("sentiment_lstm_model.keras", compile=True)
        tokenizer = joblib.load("tokenizer.pkl")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load model or tokenizer: {e}")
        return None, None

# ======================
# TEXT CLEANING
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# ======================
# PREDICTION FUNCTION
# ======================
def predict_sentiment(model, tokenizer, review):
    cleaned_review = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0]
    label = np.argmax(prediction)

    sentiment_map = {
        0: ("Negative", "😠", "red"),
        1: ("Neutral", "😐", "blue"),
        2: ("Positive", "😊", "green")
    }

    sentiment, emoji, color = sentiment_map[label]

    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "color": color,
        "probabilities": prediction,
        "confidence": float(np.max(prediction))
    }

# ======================
# MAIN INTERFACE
# ======================
st.title("📊 Sentiment Analyzer (LSTM)")
st.write("Analyze the sentiment of any coffee review using a Bidirectional LSTM model.")

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.stop()

review_input = st.text_area("📝 Enter your review below:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = predict_sentiment(model, tokenizer, review_input)
        st.markdown(f"### Sentiment: **:{result['emoji']}: {result['sentiment']}**")
        st.progress(result['confidence'], text=f"Confidence: {result['confidence']*100:.2f}%")

        st.write("**Prediction Probabilities:**")
        st.write({
            "Negative": f"{result['probabilities'][0]*100:.2f}%",
            "Neutral": f"{result['probabilities'][1]*100:.2f}%",
            "Positive": f"{result['probabilities'][2]*100:.2f}%"
        })

