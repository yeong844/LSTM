import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Title
st.title("☕ Coffee Review Sentiment Classifier (LSTM)")

# Header
st.header("1. Enter Coffee Review")
user_input = st.text_area("Type your coffee review here", "")

# Constants
MAX_LEN = 100
VOCAB_SIZE = 10000

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("sentiment_lstm_model.keras")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Predict function
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded)
    label = np.argmax(prediction)
    sentiment_map = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
    return sentiment_map[label], prediction

# Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        sentiment, probs = predict_sentiment(user_input)
        st.subheader("Sentiment:")
        st.success(sentiment)
        st.write("Prediction Probabilities:", probs)
