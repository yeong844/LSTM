import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import joblib
import os

# App Title
st.title("☕ Coffee Review Sentiment Classifier (LSTM)")

# Load Dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file with 'reviews' and 'stars' columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data:", df.head())

    # Drop missing values
    df.dropna(inplace=True)

    # Map sentiment
    def map_sentiment(stars):
        if stars <= 2:
            return 0  # Negative
        elif stars == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df['label'] = df['stars'].apply(map_sentiment)
    df = df[['reviews', 'label']]

    # Train-Test Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['reviews'], df['label'], test_size=0.2, random_state=42)

    # Tokenize and Pad
    max_len = 100
    vocab_size = 10000

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Try to load existing model
    model = None
    if os.path.exists("sentiment_lstm_model.keras") and os.path.exists("tokenizer.pkl"):
        model = tf.keras.models.load_model("sentiment_lstm_model.keras")
        tokenizer = joblib.load("tokenizer.pkl")
        st.success("✅ Pre-trained model and tokenizer loaded.")

    else:
        # Define Model (will train if user clicks button)
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
            LSTM(128),
            Dense(3, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training section
    st.header("2. Train Model")
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            history = model.fit(train_padded, train_labels, epochs=5, batch_size=16,
                                validation_data=(test_padded, test_labels), verbose=0)
            model.save("sentiment_lstm_model.keras")
            joblib.dump(tokenizer, 'tokenizer.pkl')
            st.success("✅ Training complete and model saved!")

    # Evaluation
    if model:
        st.header("3. Evaluate Model")
        preds = model.predict(test_padded)
        preds = np.argmax(preds, axis=1)

        report = classification_report(test_labels, preds, target_names=["Negative", "Neutral", "Positive"], output_dict=True)
        st.write("📊 Classification Report", pd.DataFrame(report).transpose())

        cm = confusion_matrix(test_labels, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
