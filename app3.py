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
MAX_LEN = 100  # same as training

# ======================
# MODEL & TOKENIZER LOADING
# ======================
@st.cache_resource  # Use caching to avoid reloading on every rerun
def load_model_and_tokenizer():
    try:
        model = load_model("sentiment_lstm_model.keras", compile=True)  # Ensure model is compiled
        tokenizer = joblib.load("tokenizer.pkl")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        return None, None

# ======================
# PREDICTION FUNCTION
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_sentiment(model, tokenizer, text):
    if not text.strip():
        return None

    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        prediction = model.predict(padded, verbose=0)

        label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        sentiments = ["Negative", "Neutral", "Positive"]
        emojis = ["😞", "😐", "😊"]

        return {"sentiment": sentiments[label], "emoji": emojis[label], "confidence": confidence, "probabilities": prediction[0]}

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# ======================
# MAIN INTERFACE
# ======================
st.title("Sentiment Analyzer")
st.write("Analyze the sentiment of any text review using a Bidirectional LSTM model.")

# Check model files
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
    # Only load model if files exist
    model, tokenizer = load_model_and_tokenizer()

# User input section
st.header("Enter a Review")
user_input = st.text_area(
    "Review Text:", 
    height=150, 
    value="",  # Start with empty input
    placeholder="Example: This product was decent but not impressive. Could have been better."
)

# Analysis button
analyze_button = st.button("Analyze Sentiment", type="primary")

if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    elif model is None or tokenizer is None:
        st.error("Cannot analyze: Model or tokenizer could not be loaded.")
    else:
        with st.spinner("Analyzing..."):
            # The actual prediction
            results = predict_sentiment(model, tokenizer, user_input)

            sentiment = results["sentiment"]
            emoji = results["emoji"]
            confidence = results["confidence"]
            probabilities = results["probabilities"]

            # Show main result
            st.markdown(
                f"### <span style='color:{'green' if sentiment == 'Positive' else 'blue' if sentiment == 'Neutral' else 'red'};'>{emoji} {sentiment}</span>",
                unsafe_allow_html=True
            )

            # Show confidence bar
           st.progress(int(confidence * 100))
           st.caption(f"Confidence: {confidence:.1%}")

            # Show detailed breakdown
            st.subheader("Sentiment Breakdown")
            cols = st.columns(3)
            cols[0].metric("Negative", f"{probabilities[0]:.1%}")
            cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
            cols[2].metric("Positive", f"{probabilities[2]:.1%}")

# ======================
# SIDEBAR INFORMATION
# ======================
st.sidebar.title("About")
st.sidebar.info("""
This app uses a Bidirectional LSTM model trained to classify sentiment for any text input.
The model categorizes reviews as:
- 😠 **Negative**
- 😐 **Neutral**
- 😊 **Positive**
""")

st.sidebar.subheader("Model Information")
st.sidebar.markdown("""
- **Architecture**: Bidirectional LSTM
- **Training Data**: Text data with sentiment labels
- **Features**: Text processing with NLP
- **Target**: Sentiment classification (3 classes)
""")

st.sidebar.subheader("📢 Important Notice")
st.sidebar.warning("""
**The model is currently being improved for better accuracy.**

This version includes fixes to better identify all sentiment types, 
including neutral reviews. A fully retrained model will be 
released soon.
""")

st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
1. **Be descriptive** - Include details about your experience, features, etc.
2. **Use specific terms** - Words that describe performance, quality, etc.
3. **Be clear** - Avoid overly ambiguous or vague language.
""")
