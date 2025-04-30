import pandas as pd
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# === Label mapping ===
LABEL_MAP = {
    "false": 0, "fake": 0, "FALSE": 0, "Fake": 0,
    "true": 1, "real": 1, "TRUE": 1, "True": 1
}

# === Step 1: Load and Process CSV ===
def load_and_prepare_text_data(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize labels
    df['normalized_label'] = df['label'].astype(str).str.lower().map(LABEL_MAP)
    df = df[df['normalized_label'].notnull()]

    # Select 1000 true and 1000 fake entries
    true_df = df[df['normalized_label'] == 1].head(20000)
    fake_df = df[df['normalized_label'] == 0].head(20000)

    selected_df = pd.concat([true_df, fake_df], ignore_index=True)

    # Combine title and description
    texts = selected_df['title'].fillna('') + ". " + selected_df['description'].fillna('')

    labels = selected_df['normalized_label'].values

    return texts.tolist(), labels

# === Step 2: Tokenize Text ===
def tokenize_texts(texts, vocab_size=5000, max_length=100):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return padded_sequences, tokenizer

# === Step 3: Model Definition ===
def build_rnn_model(vocab_size=5000, embedding_dim=128, input_length=100):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        SimpleRNN(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# === Full pipeline ===
csv_path = "merged_news.csv"  # Update your CSV file path
texts, labels = load_and_prepare_text_data(csv_path)

X, tokenizer = tokenize_texts(texts)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build and train model
rnn_model = build_rnn_model(vocab_size=5000, embedding_dim=128, input_length=100)
rnn_model.summary()

rnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
