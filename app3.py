import streamlit as st
import numpy as np
import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Define prediction function (yours)
def predict_sentiment(model, tokenizer, review):
    try:
        review = clean_text(review)  # Ensure same preprocessing
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
        prediction = model.predict(padded, verbose=0)
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
            "probabilities": prediction[0]
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
