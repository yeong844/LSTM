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
        st.error(f"Failed to load model or tokenizer: {e}")
        return None, None

# ======================
# PREDICTION FUNCTION (Simplified)
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

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
        "confidence": float(np.max(prediction)),
        "corrected": False,
        "original_prediction": None
    }

# ======================
# MAIN INTERFACE
# ======================
st.title("Sentiment Analyzer(LSTM)")
st.write("Analyze the sentiment of any text review using a Bidirectional LSTM model.")

file_status = st.empty()
model_exists = os.path.exists("sentiment_lstm_model.keras")
tokenizer_exists = os.path.exists("tokenizer.pkl")

if not model_exists or not tokenizer_exists:
    file_status.error("⚠️ Model or tokenizer file missing!")
    missing_files = []
    if not model_exists:
        missing_files.append("sentiment_lstm_model.keras")
    if not tokenizer_exists:
        missing_files.append("tokenizer.pkl")
    
    st.info(f"""
    Please ensure these files are in the app directory:
    - {', '.join(missing_files)}
    """)
else:
    file_status.success("✅ Model and tokenizer files found")
    model, tokenizer = load_model_and_tokenizer()

# Input section
st.header("Sentiment Analyzer(LSTM)")
user_input = st.text_area(
    "Review Text:",
    height=150,
    value="",
    placeholder="Example: The product quality was great and delivery was fast!"
)

analyze_button = st.button("Analyze Sentiment", type="primary")

if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    elif model is None or tokenizer is None:
        st.error("Cannot analyze: Model or tokenizer could not be loaded.")
    else:
        with st.spinner("Analyzing..."):
            results = predict_sentiment(model, tokenizer, user_input)

            sentiment = results["sentiment"]
            emoji = results["emoji"]
            color = results["color"]
            probabilities = results["probabilities"]
            confidence = results["confidence"]

            st.markdown(
                f"### <span style='color:{color}; font-size: 28px;'>{emoji} {sentiment}</span>",
                unsafe_allow_html=True
            )

            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")

            st.subheader("Sentiment Breakdown")
            cols = st.columns(3)
            cols[0].metric("Negative", f"{probabilities[0]:.1%}")
            cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
            cols[2].metric("Positive", f"{probabilities[2]:.1%}")

            with st.expander("Preprocessing Details"):
                st.write("**Original Text:**")
                st.write(user_input)
                st.write("**Processed Text:**")
                st.write(clean_text(user_input))

# ======================
# SIDEBAR INFORMATION
# ======================
st.sidebar.title("About")
st.sidebar.info("""
This app uses a Bidirectional LSTM model trained to classify sentiment for any text input.
Sentiment Categories:
- 😠 Negative
- 😐 Neutral
- 😊 Positive
""")

st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
1. Use complete sentences.
2. Mention what you liked or disliked.
3. Avoid vague language.
""")
