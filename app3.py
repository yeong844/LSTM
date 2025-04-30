import streamlit as st
import numpy as np
import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# CONSTANTS
# ======================
MAX_LEN = 100
VOCAB_SIZE = 10000

# ======================
# CLEANING FUNCTION
# ======================
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# ======================
# LOAD MODEL AND TOKENIZER
# ======================
def load_model_and_tokenizer():
    try:
        model = load_model("sentiment_lstm_model.keras")
        tokenizer = joblib.load("tokenizer.pkl")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        return None, None

# ======================
# PREDICTION FUNCTION
# ======================
def predict_sentiment(model, tokenizer, review):
    try:
        # Clean the review text
        review = clean_text(review)

        # Convert the review text to sequence of integers
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

        # Predict sentiment
        prediction = model.predict(padded, verbose=0)
        
        # Get the predicted class
        label = np.argmax(prediction)
        
        # Map label to sentiment
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

# ======================
# STREAMLIT PAGE SETUP
# ======================
try:
    st.set_page_config(
        page_title="Coffee Review Sentiment (LSTM)",
        layout="centered",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Page configuration error: {e}")

# ======================
# MAIN INTERFACE
# ======================
st.title("☕ Coffee Review Sentiment Classifier (LSTM)")
st.write("Analyze the sentiment of a coffee review using a pre-trained LSTM model.")

st.header("Enter a Coffee Review")
user_input = st.text_area("Review Text:", height=150)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            model, tokenizer = load_model_and_tokenizer()
            if model is not None and tokenizer is not None:
                results = predict_sentiment(model, tokenizer, user_input)

                if results:
                    sentiment = results["sentiment"]
                    emoji = results["emoji"]
                    color = results["color"]
                    probabilities = results["probabilities"]

                    st.markdown(
                        f"### <span style='color:{color}'>{emoji} {sentiment}</span>",
                        unsafe_allow_html=True
                    )

                    st.progress(int(np.max(probabilities) * 100))
                    st.caption(f"Confidence: {np.max(probabilities):.1%}")

                    with st.expander("Detailed Prediction"):
                        cols = st.columns(3)
                        cols[0].metric("Positive", f"{probabilities[2]:.1%}")
                        cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
                        cols[2].metric("Negative", f"{probabilities[0]:.1%}")

# ======================
# SIDEBAR NOTICE
# ======================
st.sidebar.warning("""
⚠️ **Security Warning**  
This app loads models and tokenizers from files that may execute arbitrary code.  
Only use trusted model/tokenizer files.
""")
