import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tempfile
import os

# ======================
# STREAMLIT INITIALIZATION
# ======================
try:
    st.set_page_config(
        page_title="Sentiment Analysis (LSTM)",
        layout="centered",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Initialization error: {e}")

# ======================
# UPLOAD SECTION
# ======================
st.sidebar.header("Upload Required Files")
model_file = st.sidebar.file_uploader("Upload Keras Model (.keras)", type=["keras"])
tokenizer_file = st.sidebar.file_uploader("Upload Tokenizer (.pkl)", type=["pkl"])

# ======================
# MODEL & TOKENIZER LOADING
# ======================
@st.cache_resource
def load_model_and_tokenizer(model_file, tokenizer_file):
    try:
        # Save uploaded files to temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_token:
            tmp_token.write(tokenizer_file.read())
            token_path = tmp_token.name

        model = load_model(model_path)

        with open(token_path, 'rb') as f:
            tokenizer = pickle.load(f)

        return model, tokenizer
    except Exception as e:
        st.error(f"Loading model/tokenizer failed: {str(e)}")
        return None, None

# ======================
# PREDICTION FUNCTION
# ======================
max_len = 150  # same as training

def predict_sentiment(model, tokenizer, text):
    try:
        if not text.strip():
            return None

        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_len)
        prediction = model.predict(padded, verbose=0)

        label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        sentiments = ["Negative", "Neutral", "Positive"]
        emojis = ["😞", "😐", "😊"]

        return {
            "sentiment": sentiments[label],
            "emoji": emojis[label],
            "confidence": confidence,
            "probabilities": prediction[0]
        }

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# ======================
# MAIN APP INTERFACE
# ======================
st.title("Review Sentiment Analysis (LSTM)")
st.write("Enter your review below:")

user_input = st.text_area("Review Text:", height=150)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    elif not model_file or not tokenizer_file:
        st.warning("⚠️ Please upload both the model and tokenizer files in the sidebar.")
    else:
        with st.spinner("Processing..."):
            model, tokenizer = load_model_and_tokenizer(model_file, tokenizer_file)

            if model is not None and tokenizer is not None:
                results = predict_sentiment(model, tokenizer, user_input)

                if results:
                    sentiment = results["sentiment"]
                    emoji = results["emoji"]
                    confidence = results["confidence"]
                    probabilities = results["probabilities"]

                    # Color Mapping
                    color_map = {
                        "Positive": "green",
                        "Neutral": "blue",
                        "Negative": "red"
                    }
                    color = color_map.get(sentiment, "gray")

                    st.markdown(
                        f"### <span style='color:{color}'>{emoji} {sentiment}</span>",
                        unsafe_allow_html=True
                    )

                    st.progress(int(confidence * 100))
                    st.caption(f"Confidence: {confidence:.1%}")

                    with st.expander("Detailed Analysis"):
                        cols = st.columns(3)
                        cols[0].metric("Positive", f"{probabilities[2]:.1%}")
                        cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
                        cols[2].metric("Negative", f"{probabilities[0]:.1%}")

# ======================
# TROUBLESHOOTING SECTION
# ======================
with st.expander("⚠️ Troubleshooting Help"):
    st.markdown("""
    **Common Issues & Solutions:**

    1. **Model or Tokenizer not found**:
       - Upload `sentiment_lstm_model.keras` and `tokenizer.pkl` using the sidebar.
       - Refresh the page after uploading.

    2. **Strange predictions**:
       - Input longer and more explicit reviews for better results.

    3. **App crashes**:
       - Restart the Streamlit server.
       - Check console logs for detailed errors.
    """)

# ======================
# SECURITY WARNING
# ======================
st.sidebar.warning("""
⚠️ **Security Notice**
This app loads models and tokenizers from pickle/keras files which can execute arbitrary code.
Only use models from trusted sources.
""")
