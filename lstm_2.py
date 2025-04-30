# Install dependencies if not already installed
# !pip install pandas scikit-learn tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# === Label mapping ===
LABEL_MAP = {
    "false": 0, "fake": 0, "FALSE": 0, "Fake": 0,
    "true": 1, "real": 1, "TRUE": 1, "True": 1
}

# === Step 1: Load and Prepare Data ===
def load_and_prepare_text_data(csv_path):
    # Read CSV with fallback for malformed rows
    print("Loading data from", csv_path)
    df = pd.read_csv(csv_path, engine='python', quoting=3, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows")

    # Normalize labels
    df['normalized_label'] = df['label'].astype(str).str.lower().map(LABEL_MAP)
    df = df[df['normalized_label'].notnull()]  # Remove rows with unknown labels
    print(f"After label normalization: {len(df)} rows")

    # Select data samples (limited to 20000 of each class for original dataset)
    true_df = df[df['normalized_label'] == 1].head(20000)
    fake_df = df[df['normalized_label'] == 0].head(20000)
    selected_df = pd.concat([true_df, fake_df], ignore_index=True)
    print(f"Selected {len(selected_df)} rows ({len(true_df)} true, {len(fake_df)} fake)")

    # Combine title + description
    texts = selected_df['title'].fillna('') + ". " + selected_df['description'].fillna('')
    labels = selected_df['normalized_label'].values

    return texts.tolist(), labels

# === Step 2: Tokenization and Padding ===
def tokenize_and_pad(texts, vocab_size=5000, max_length=100):
    print(f"Tokenizing with vocab size {vocab_size} and max length {max_length}")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    print(f"Padded shape: {padded.shape}")

    return padded, tokenizer

# === Step 3: Set up and run the pipeline ===
# Update to your actual file path
csv_path = "merged_news.csv"
texts, labels = load_and_prepare_text_data(csv_path)

X, tokenizer = tokenize_and_pad(texts, vocab_size=5000, max_length=100)
y = np.array(labels)

# Train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Reduce training data by 50% to reduce accuracy
sample_size = len(X_train_full) // 2
print(f"Reducing training data from {len(X_train_full)} to {sample_size} samples")

# Use random sampling to create a reduced training set
np.random.seed(42)
sampled_indices = np.random.choice(len(X_train_full), sample_size, replace=False)
X_train = X_train_full[sampled_indices]
y_train = y_train_full[sampled_indices]

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# === Step 4: Build LSTM Model ===
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),  # Reduced from 128
    LSTM(32, return_sequences=True),  # Reduced complexity
    Dropout(0.5),
    LSTM(16),  # Reduced from 32/64
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model summary:")
model.summary()

# === Step 5: Train Model ===
print("Training model on reduced dataset...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=1
)

# === Step 6: Evaluate Model ===
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Final test accuracy: {test_accuracy:.4f}")
print(f"Final test loss: {test_loss:.4f}")

# === Step 7: Plot Training History ===
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy with Reduced Training Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss with Reduced Training Data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# === Step 8: Make predictions on test data ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate accuracy manually
accuracy = np.mean(y_pred == y_test)
print(f"Manual accuracy check: {accuracy:.4f}")

# Examine misclassifications
misclassified_indices = np.where(y_pred != y_test)[0]
print(f"Number of misclassifications: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test):.2%})")

# Save the model if needed
model.save('fake_news_reduced_accuracy_model.h5')
print("Model saved to fake_news_reduced_accuracy_model.h5")