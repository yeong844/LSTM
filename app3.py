import streamlit as st
import numpy as np
import joblib
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('sentiment_lstm_model.keras')
tokenizer = joblib.load('tokenizer.pkl')

# Define cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Define prediction function
def predict_sentiment(model, tokenizer, review):
    try:
        review = clean_text(review)  # Ensure same preprocessing
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=100, padding='post')
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

# Streamlit interface
st.title('Sentiment Analysis App')

st.write("Enter a review to analyze its sentiment:")

# Input from user
user_review = st.text_area("Review", "")

if user_review:
    # Predict sentiment
    result = predict_sentiment(model, tokenizer, user_review)
    
    if result:
        sentiment = result['sentiment']
        emoji = result['emoji']
        color = result['color']
        probabilities = result['probabilities']
        
        st.markdown(f"**Sentiment:** {sentiment} {emoji}")
        st.markdown(f"**Probability (Negative, Neutral, Positive):** {probabilities}")
        st.markdown(f"<h3 style='color:{color}'>Sentiment Result: {sentiment}</h3>", unsafe_allow_html=True)
