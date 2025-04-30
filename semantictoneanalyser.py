import spacy
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# === Feature Extraction Functions ===
def get_semantic_vector(text):
    doc = nlp(text)
    return doc.vector

def get_tone_scores(text):
    scores = analyzer.polarity_scores(text)
    return np.array(list(scores.values()))  # [neg, neu, pos, compound]

def get_combined_features(semantic_vec, tone_vec):
    return np.concatenate([semantic_vec, tone_vec])

# === Label Mapping ===
LABEL_MAP = {
    "false": 0, "fake": 0, "FALSE": 0, "Fake": 0,
    "true": 1, "real": 1, "TRUE": 1, "True": 1
}

# === Main Processing Function ===
def process_and_save_separately(input_csv_path, out_prefix="output"):
    df = pd.read_csv(input_csv_path)

    # Normalize labels
    df['normalized_label'] = df['label'].astype(str).str.lower().map(LABEL_MAP)
    df = df[df['normalized_label'].notnull()]

    # Take 1000 true and 1000 fake
    true_df = df[df['normalized_label'] == 1].head(10000)
    fake_df = df[df['normalized_label'] == 0].head(10000)
    selected_df = pd.concat([true_df, fake_df], ignore_index=True)

    print(f"✅ Selected {len(fake_df)} fake and {len(true_df)} true entries.")

    semantic_features = []
    tone_features = []
    combined_features = []
    labels = []

    for idx, row in selected_df.iterrows():
        title = str(row.get('title', ''))
        desc = str(row.get('description', ''))
        full_text = f"{title}. {desc}"

        sem_vec = get_semantic_vector(full_text)
        tone_vec = get_tone_scores(full_text)
        comb_vec = get_combined_features(sem_vec, tone_vec)

        semantic_features.append(sem_vec)
        tone_features.append(tone_vec)
        combined_features.append(comb_vec)
        labels.append(int(row['normalized_label']))

    # Convert to DataFrames
    semantic_df = pd.DataFrame(semantic_features, columns=[f'sem_{i}' for i in range(len(semantic_features[0]))])
    tone_df = pd.DataFrame(tone_features, columns=['tone_neg', 'tone_neu', 'tone_pos', 'tone_compound'])
    combined_df = pd.DataFrame(combined_features, columns=[f'comb_{i}' for i in range(len(combined_features[0]))])

    label_df = pd.DataFrame({'label': labels})

    # Add labels
    semantic_df['label'] = labels
    tone_df['label'] = labels
    combined_df['label'] = labelsza

    # Save CSVs
    semantic_df.to_csv(f"{out_prefix}_semantic.csv", index=False)
    tone_df.to_csv(f"{out_prefix}_tone.csv", index=False)
    combined_df.to_csv(f"{out_prefix}_combined.csv", index=False)

    print("✅ Saved:")
    print(f"    {out_prefix}_semantic.csv → {semantic_df.shape}")
    print(f"    {out_prefix}_tone.csv     → {tone_df.shape}")
    print(f"    {out_prefix}_combined.csv → {combined_df.shape}")

# === Example Usage ===
process_and_save_separately("merged_news.csv", out_prefix="output_10000_true_10000_fake")