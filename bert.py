import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# === Load and prepare text data ===
def load_and_prepare_text_data(csv_path):
    df = pd.read_csv(csv_path, engine='python', quoting=3, on_bad_lines='skip')

    # Label mapping
    LABEL_MAP = {
        "false": 0, "fake": 0, "FALSE": 0, "Fake": 0,
        "true": 1, "real": 1, "TRUE": 1, "True": 1
    }
    df['normalized_label'] = df['label'].astype(str).str.lower().map(LABEL_MAP)
    df = df[df['normalized_label'].notnull()]

    # Select first 10000 true and 10000 fake entries
    true_df = df[df['normalized_label'] == 1].head(10000)
    fake_df = df[df['normalized_label'] == 0].head(10000)

    selected_df = pd.concat([true_df, fake_df], ignore_index=True)

    texts = selected_df['title'].astype(str) + " " + selected_df['description'].astype(str)
    labels = selected_df['normalized_label'].values

    print(f"✅ Selected {len(texts)} texts.")

    return texts.tolist(), labels

# === BERT Tokenizer ===
def encode_texts(texts, tokenizer):
    return tokenizer(
        texts, padding=True, truncation=True, return_tensors="tf", max_length=512
    )

# === Main Code ===
csv_path = "merged_news.csv"  # Your file

texts, labels = load_and_prepare_text_data(csv_path)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

train_encodings = encode_texts(X_train, bert_tokenizer)
test_encodings = encode_texts(X_test, bert_tokenizer)

# Define Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

# Compile model
bert_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
bert_model.fit(
    x={
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    },
    y=y_train,
    validation_data=(
        {
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask']
        },
        y_test
    ),
    epochs=3,
    batch_size=16
)