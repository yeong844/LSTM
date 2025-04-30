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
    page_title="Coffee Review Sentiment (LSTM)",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================
# CONSTANTS
# ======================
MAX_LEN = 100
VOCAB_SIZE = 10000

# ======================
# MODEL & TOKENIZER LOADING
# ======================
@st.cache_resource  # Use caching to avoid reloading on every rerun
def load_model_and_tokenizer():
    try:
        model = load_model("sentiment_lstm_model.keras")
        tokenizer = joblib.load("tokenizer.pkl")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        st.info("Make sure the model and tokenizer files are in the same directory as this app.")
        return None, None

# ======================
# PREDICTION FUNCTION
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def predict_sentiment(model, tokenizer, review):
    try:
        # Clean and preprocess the review
        cleaned_review = clean_text(review)
        
        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([cleaned_review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)
        
        # Get the predicted class
        label = np.argmax(prediction[0])
        
        # Map the prediction to sentiment
        sentiment_map = {
            0: ("Negative", "😠", "red"),
            1: ("Neutral", "😐", "blue"),
            2: ("Positive", "😊", "green")
        }
        
        sentiment, emoji, color = sentiment_map[label]
        
        # Debug information
        st.write(f"Debug - Padded sequence shape: {padded.shape}")
        st.write(f"Debug - Raw prediction values: {prediction[0]}")
        
        return {
            "sentiment": sentiment,
            "emoji": emoji,
            "color": color,
            "probabilities": prediction[0]
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write(f"Error details: {str(e)}")
        return None

# ======================
# MAIN INTERFACE
# ======================
st.title("☕ Coffee Review Sentiment Classifier (LSTM)")
st.write("Analyze the sentiment of a coffee review using a pre-trained LSTM model.")

# Check model files
model_exists = os.path.exists("sentiment_lstm_model.keras")
tokenizer_exists = os.path.exists("tokenizer.pkl")

if not model_exists or not tokenizer_exists:
    st.error("⚠️ Model or tokenizer file missing!")
    st.info("""
    Please ensure both files are in the app directory:
    - sentiment_lstm_model.keras
    - tokenizer.pkl
    """)
else:
    st.success("✅ Model and tokenizer files found")

st.header("Enter a Coffee Review")
user_input = st.text_area("Review Text:", height=150, 
                          placeholder="Example: This coffee has amazing flavor with hints of chocolate and a smooth finish.")

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            model, tokenizer = load_model_and_tokenizer()
            
            if model is not None and tokenizer is not None:
                # Print some debugging info
                st.write(f"Model summary: {model.summary()}")
                st.write(f"Tokenizer word index size: {len(tokenizer.word_index)}")
                
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
                    
                    st.progress(float(np.max(probabilities)))
                    st.caption(f"Confidence: {np.max(probabilities):.1%}")
                    
                    with st.expander("Detailed Prediction", expanded=True):
                        cols = st.columns(3)
                        cols[0].metric("Negative", f"{probabilities[0]:.1%}")
                        cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
                        cols[2].metric("Positive", f"{probabilities[2]:.1%}")

                    # Add example reviews for testing
                    st.subheader("Try these example reviews:")
                    examples = {
                        "Positive": "This coffee is amazing! Rich flavor with hints of chocolate. Best I've ever had!",
                        "Neutral": "The coffee is okay. Not the best, not the worst. It's drinkable but nothing special.",
                        "Negative": "Terrible coffee! Tastes burnt and bitter. I couldn't even finish my cup."
                    }
                    
                    for sentiment, example in examples.items():
                        if st.button(f"Try {sentiment} Example"):
                            st.session_state.user_input = example
                            st.experimental_rerun()

# ======================
# SIDEBAR INFORMATION
# ======================
st.sidebar.title("About")
st.sidebar.info("""
This app uses a Bidirectional LSTM model trained on coffee review data to predict sentiment.
The model classifies reviews as Negative, Neutral, or Positive based on text content.
""")

st.sidebar.warning("""
⚠️ **Security Warning**  
This app loads models and tokenizers from files that may execute arbitrary code.  
Only use trusted model/tokenizer files.
""")

st.sidebar.subheader("Troubleshooting")
st.sidebar.markdown("""
If predictions seem stuck or incorrect:
1. Make sure the model and tokenizer files are correctly loaded
2. Check that the text preprocessing matches what was used during training
3. Try restarting the app
""")
