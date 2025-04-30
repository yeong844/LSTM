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
    page_title="Coffee Review Sentiment Analyzer",
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
    # Match exact preprocessing from training
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_sentiment(model, tokenizer, review):
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
    
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "color": color,
        "probabilities": prediction[0],
        "confidence": float(np.max(prediction[0]))
    }

# ======================
# MAIN INTERFACE
# ======================
st.title("☕ Coffee Review Sentiment Analyzer")
st.write("Analyze the sentiment of coffee reviews using a Bidirectional LSTM model.")

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
st.header("Enter a Coffee Review")
user_input = st.text_area(
    "Review Text:", 
    height=150, 
    value="",  # Start with empty input
    placeholder="Example: This coffee has amazing flavor with hints of chocolate and a smooth finish."
)

# Example section
with st.expander("Try example reviews"):
    examples = {
        "Positive": "This coffee is amazing! Rich flavor with hints of chocolate. Best I've ever had!",
        "Neutral": "The coffee is okay. Not the best, not the worst. It's drinkable but nothing special.",
        "Negative": "Terrible coffee! Tastes burnt and bitter. I couldn't even finish my cup."
    }
    
    cols = st.columns(3)
    for i, (sentiment, example) in enumerate(examples.items()):
        if cols[i].button(f"{sentiment} Example"):
            user_input = example
            st.session_state.user_input = example

# Analysis button
analyze_button = st.button("Analyze Sentiment", type="primary")

# Handle session state for examples
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input
    analyze_button = True

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
            color = results["color"]
            probabilities = results["probabilities"]
            confidence = results["confidence"]
            
            # Show main result
            st.markdown(
                f"### <span style='color:{color}; font-size: 28px;'>{emoji} {sentiment}</span>",
                unsafe_allow_html=True
            )
            
            # Show confidence bar
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            # Show detailed breakdown
            st.subheader("Sentiment Breakdown")
            cols = st.columns(3)
            cols[0].metric("Negative", f"{probabilities[0]:.1%}")
            cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
            cols[2].metric("Positive", f"{probabilities[2]:.1%}")
            
            # Show the processed text
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
This app uses a Bidirectional LSTM model trained on coffee review data to classify sentiment.
The model categorizes reviews as:
- 😠 **Negative** (1-2 stars)
- 😐 **Neutral** (3 stars)
- 😊 **Positive** (4-5 stars)
""")

st.sidebar.subheader("Model Information")
st.sidebar.markdown("""
- **Architecture**: Bidirectional LSTM
- **Training Data**: Coffee reviews with star ratings
- **Features**: Text reviews processed with NLP
- **Target**: Sentiment classification (3 classes)
""")

st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
1. **Be descriptive** - Include details about flavor, aroma, etc.
2. **Use coffee terminology** - Words like "acidic," "bitter," or "smooth"
3. **Be clear** - Avoid ambiguous language
""")

st.sidebar.subheader("Troubleshooting")
if model is not None and tokenizer is not None:
    st.sidebar.success("Model and tokenizer loaded successfully")
else:
    st.sidebar.error("Model or tokenizer could not be loaded")
    st.sidebar.markdown("""
    If you're encountering issues:
    1. Check that model files are in the same directory as this app
    2. Restart the application
    3. Try reinstalling the model files
    """)
