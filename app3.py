import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json

# Load model and tokenizer from .pkl files
@st.cache_resource
def load_model_and_tokenizer():
    model_dict = joblib.load("sentiment_lstm_model.pkl")
    tokenizer = joblib.load("tokenizer.pkl")

    model = model_from_json(model_dict["model"])
    model.set_weights(model_dict["weights"])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Streamlit interface
st.title("Coffee Review Sentiment Analyzer")

review = st.text_area("Enter your coffee review here:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)

        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.write("### Sentiment:", sentiment_labels[predicted_class])
        st.write(f"Confidence Score: {np.max(prediction):.2f}")
