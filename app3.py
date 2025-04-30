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
    
    # Define sentiment keywords
    negative_keywords = ['terrible', 'awful', 'bad', 'worst', 'horrible', 'disgusting', 
                        'bitter', 'burnt', 'hate', 'nasty', 'undrinkable', 'disappointed']
    
    neutral_keywords = ['okay', 'ok', 'average', 'decent', 'mediocre', 'moderate', 
                       'fair', 'middle', 'ordinary', 'standard', 'acceptable', 'alright',
                       'not bad', 'not great']
    
    positive_keywords = ['excellent', 'amazing', 'love', 'delicious', 'great', 'best', 
                         'fantastic', 'wonderful', 'perfect', 'awesome', 'superb', 'outstanding']
    
    # Check for presence of sentiment keywords
    has_negative = any(word in cleaned_review for word in negative_keywords)
    has_neutral = any(word in cleaned_review for word in neutral_keywords) or ('not bad' in cleaned_review and 'not great' in cleaned_review)
    has_positive = any(word in cleaned_review for word in positive_keywords)
    
    # Count keyword matches
    negative_count = sum(1 for word in negative_keywords if word in cleaned_review)
    neutral_count = sum(1 for word in neutral_keywords if word in cleaned_review)
    positive_count = sum(1 for word in positive_keywords if word in cleaned_review)
    
    # Override probabilities if clear keywords are present and prediction contradicts
    original_prediction = prediction[0].copy()
    original_class = np.argmax(original_prediction)
    corrected = False
    
    # Define thresholds for correction
    confidence_threshold = 0.6
    
    # Handle neutral cases specifically
    if has_neutral and not (has_positive or has_negative) and original_class != 1:
        # Enhance neutral probability and reduce others
        prediction[0][1] = max(prediction[0][1], confidence_threshold)  # Boost neutral
        prediction[0][0] = min(prediction[0][0], (1 - prediction[0][1]) / 2)  # Reduce negative
        prediction[0][2] = min(prediction[0][2], (1 - prediction[0][1]) / 2)  # Reduce positive
        corrected = True
    
    # Mixed sentiment with more neutral keywords than others
    elif neutral_count > positive_count and neutral_count > negative_count and original_class != 1:
        prediction[0][1] = max(prediction[0][1], confidence_threshold)  # Boost neutral
        prediction[0][0] = min(prediction[0][0], (1 - prediction[0][1]) / 2)  # Reduce negative
        prediction[0][2] = min(prediction[0][2], (1 - prediction[0][1]) / 2)  # Reduce positive
        corrected = True
    
    # If clear negative keywords but predicted positive
    elif has_negative and not has_positive and original_class == 2:
        # Swap probabilities (negative and positive)
        prediction[0][0] = max(original_prediction[2], confidence_threshold)  # Set negative high
        prediction[0][2] = min(original_prediction[0], (1 - prediction[0][0] - prediction[0][1]))  # Reduce positive
        corrected = True
    
    # If clear positive keywords but predicted negative
    elif has_positive and not has_negative and original_class == 0:
        # Swap probabilities (negative and positive)
        prediction[0][2] = max(original_prediction[0], confidence_threshold)  # Set positive high
        prediction[0][0] = min(original_prediction[2], (1 - prediction[0][1] - prediction[0][2]))  # Reduce negative
        corrected = True
    
    # Normalize probabilities to sum to 1
    prediction[0] = prediction[0] / np.sum(prediction[0])
    
    # Get the predicted class after possible correction
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
        "confidence": float(np.max(prediction[0])),
        "corrected": corrected,
        "original_prediction": original_prediction if corrected else None
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
            corrected = results["corrected"]
            
            # Show main result
            st.markdown(
                f"### <span style='color:{color}; font-size: 28px;'>{emoji} {sentiment}</span>",
                unsafe_allow_html=True
            )
            
            # Show correction notice if applicable
            if corrected:
                st.warning("⚠️ Prediction was adjusted based on sentiment keywords in text.")
                st.write("The model prediction appeared to contradict clear sentiment indicators in your text.")
            
            # Show confidence bar
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            # Show detailed breakdown
            st.subheader("Sentiment Breakdown")
            cols = st.columns(3)
            cols[0].metric("Negative", f"{probabilities[0]:.1%}")
            cols[1].metric("Neutral", f"{probabilities[1]:.1%}")
            cols[2].metric("Positive", f"{probabilities[2]:.1%}")
            
            # If corrected, show original prediction
            if corrected and results["original_prediction"] is not None:
                with st.expander("Original Model Prediction"):
                    orig_pred = results["original_prediction"]
                    st.write("This was the model's original prediction before adjustment:")
                    orig_cols = st.columns(3)
                    orig_cols[0].metric("Negative", f"{orig_pred[0]:.1%}")
                    orig_cols[1].metric("Neutral", f"{orig_pred[1]:.1%}")
                    orig_cols[2].metric("Positive", f"{orig_pred[2]:.1%}")
            
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

st.sidebar.subheader("📢 Important Notice")
st.sidebar.warning("""
**The model is currently being improved for better accuracy.**

This version includes fixes to better identify all sentiment types, 
including neutral reviews. A fully retrained model will be 
released soon.
""")

st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
1. **Be descriptive** - Include details about flavor, aroma, etc.
2. **Use coffee terminology** - Words like "acidic," "bitter," or "smooth"
3. **Be clear** - Avoid ambiguous language
""")
