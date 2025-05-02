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
# PREDICTION FUNCTION (Improved)
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_sentiment(model, tokenizer, review):
    cleaned_review = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Get raw prediction probabilities
    prediction = model.predict(padded, verbose=0)[0]
    
    # Map predictions to sentiment categories (0: Negative, 1: Neutral, 2: Positive)
    label = np.argmax(prediction)
    
    sentiment_map = {
        0: ("Negative", "😠", "red"),
        1: ("Neutral", "😐", "blue"),
        2: ("Positive", "😊", "green")
    }

    sentiment, emoji, color = sentiment_map[label]
    
    # Calculate confidence as the highest probability
    confidence = float(np.max(prediction))
    
    # Apply threshold correction for more accurate sentiment classification
    # This helps handle cases where the model might be uncertain
    threshold = 0.5  # Minimum confidence threshold
    
    corrected = False
    original_prediction = None
    
    # If confidence is too low, consider it neutral
    if confidence < threshold and label != 1:  # If not already neutral
        original_prediction = label
        label = 1  # Set to neutral
        sentiment, emoji, color = sentiment_map[label]
        corrected = True
    
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "color": color,
        "probabilities": prediction,
        "confidence": confidence,
        "corrected": corrected,
        "original_prediction": original_prediction
    }

# ======================
# MAIN INTERFACE
# ======================
st.title("Sentiment Analyzer")
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
st.header("Enter a Review")
user_input = st.text_area(
    "Review Text:",
    height=150,
    value="",
    placeholder="Example: The product quality was great and delivery was fast!"
)

# Add examples for user to try
examples = [
    "This coffee is amazing with rich flavor and aroma. Absolutely loved it!",
    "The coffee is decent, not great but not bad either.",
    "This coffee tastes horrible. Very bitter and burnt flavor."
]

if st.checkbox("Show example reviews"):
    example_select = st.selectbox("Select an example:", examples)
    if st.button("Use this example"):
        user_input = example_select
        st.session_state.user_input = user_input

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
            corrected = results["corrected"]
            original_prediction = results["original_prediction"]

            st.markdown(
                f"### <span style='color:{color}; font-size: 28px;'>{emoji} {sentiment}</span>",
                unsafe_allow_html=True
            )

            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            # Show correction notice if applicable
            if corrected:
                original_sentiment = ["Negative", "Neutral", "Positive"][original_prediction]
                st.info(f"⚠️ Low confidence prediction ({confidence:.1%}). " 
                        f"Original prediction was {original_sentiment}, but I've classified it as Neutral.")

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
                
                # Show token information for debugging
                sequence = tokenizer.texts_to_sequences([clean_text(user_input)])
                padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
                
                st.write("**Tokenized Words:**")
                words = clean_text(user_input).split()
                tokens = [tokenizer.word_index.get(word, 0) for word in words]
                token_dict = {word: (token if token > 0 else "<OOV>") for word, token in zip(words, tokens)}
                st.write(token_dict)
                
                st.write("**Sequence Length:**", len(sequence[0]))
                if len(sequence[0]) == 0:
                    st.warning("⚠️ No words were recognized by the tokenizer!")

# ======================
# SIDEBAR INFORMATION
# ======================
st.sidebar.title("About")
st.sidebar.info("""
This app uses a Bidirectional LSTM model trained to classify sentiment for any text input.
Sentiment Categories:
- 😠 Negative (Star rating ≤ 2)
- 😐 Neutral (Star rating = 3)
- 😊 Positive (Star rating ≥ 4)
""")

st.sidebar.subheader("Model Details")
st.sidebar.markdown("""
- **Model**: Bidirectional LSTM
- **Tokenizer**: Trained on coffee reviews dataset
- **Input**: Any review or feedback text
- **Accuracy**: Classification is based on probabilities for each sentiment class
""")

st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
1. Use complete sentences with proper context.
2. Mention specific aspects (flavor, quality, service, etc.).
3. Be clear about what you liked or disliked.
4. Use descriptive language (e.g., "rich", "bitter", "smooth").
5. Longer, more detailed reviews tend to give better results.
""")
