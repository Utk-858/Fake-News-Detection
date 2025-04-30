# Install dependencies if not already installed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Label mapping ===
LABEL_MAP = {
    "false": 0, "fake": 0, "FALSE": 0, "Fake": 0,
    "true": 1, "real": 1, "TRUE": 1, "True": 1
}

# === Step 1: Load and Prepare Data ===
def load_and_prepare_text_data(csv_path):
    # Read CSV with fallback for malformed rows
    df = pd.read_csv(csv_path, engine='python', quoting=3, on_bad_lines='skip')

    # Normalize labels
    df['normalized_label'] = df['label'].astype(str).str.lower().map(LABEL_MAP)
    df = df[df['normalized_label'].notnull()]  # Remove rows with unknown labels

    # Select 1000 samples each
    true_df = df[df['normalized_label'] == 1].head(20000)
    fake_df = df[df['normalized_label'] == 0].head(20000)
    selected_df = pd.concat([true_df, fake_df], ignore_index=True)

    # Combine title + description
    texts = selected_df['title'].fillna('') + ". " + selected_df['description'].fillna('')
    labels = selected_df['normalized_label'].values

    return texts.tolist(), labels

# === Step 2: Tokenization and Padding ===
def tokenize_and_pad(texts, vocab_size=5000, max_length=100):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return padded, tokenizer

# === Step 3: Run the full pipeline ===
csv_path = "merged_news.csv"  # <-- Update to your actual file path
texts, labels = load_and_prepare_text_data(csv_path)

X, tokenizer = tokenize_and_pad(texts, vocab_size=5000, max_length=100)
y = np.array(labels)

# Train/test split
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# === Step 4: Build LSTM Model ===
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()

# === Step 5: Train Model ===
lstm_model.fit(X_train_nn, y_train_nn, epochs=10, validation_data=(X_test_nn, y_test_nn))
