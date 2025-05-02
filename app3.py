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
LABELS = ["Negative", "Neutral", "Positive"]

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
# TEXT CLEANING FUNCTION
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ======================
# PREDICTION FUNCTION
# ======================
def predict_sentiment(model, tokenizer, review):
    cleaned = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
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
st.markdown("Enter a product review below and let the model analyze its **sentiment**.")

model, tokenizer = load_model_and_tokenizer()

if model is not None and tokenizer is not None:
    user_input = st.text_area("✍️ Enter your review here:", height=150)
    
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review before analyzing.")
        else:
            result = predict_sentiment(model, tokenizer, user_input)
            st.markdown(
                f"<h3 style='color:{result['color']}'>Prediction: {result['sentiment']} {result['emoji']} "
                f"({result['confidence']*100:.2f}% confidence)</h3>", unsafe_allow_html=True
            )
            st.markdown("**Probability Distribution:**")
            for i, label in enumerate(LABELS):
                st.progress(float(result["probabilities"][i]), text=f"{label}: {result['probabilities'][i]*100:.2f}%")
else:
    st.error("Model and tokenizer are required to run this app. Please make sure the following files exist in the directory:")
    st.code("sentiment_lstm_model.keras")
    st.code("tokenizer.pkl")
