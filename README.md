
# 📰 Fake News Detection using Machine Learning & NLP

This project aims to build robust models to detect **fake news articles** using a combination of **classical machine learning** and **deep learning** methods, including semantic understanding of news content.

We use a variety of techniques such as:
- **Naive Bayes**
- **LSTM (Long Short-Term Memory)**
- **RNN (Recurrent Neural Networks)**
- **BERT (Bidirectional Encoder Representations from Transformers)**
- **Semantic Analysis**

---

## 📂 Dataset

- File: `merged_news.csv`
- Format: CSV
- Key columns:
  - `title`: Headline of the news
  - `description`: Short summary or description
  - `label`: `real` or `fake`
  - Additional metadata: source, subject, timestamps, etc.

---

## ⚙️ Technologies & Libraries

- **Python 3.x**
- **Pandas, NumPy**
- **Scikit-learn**
- **TensorFlow / PyTorch**
- **Transformers (HuggingFace)**
- **NLTK / spaCy** (for preprocessing)
- **Matplotlib / Seaborn** (optional for visualization)

---

## 🧠 Models Implemented

| Model             | Description |
|------------------|-------------|
| **Naive Bayes**  | Simple and effective probabilistic model using TF-IDF features. |
| **RNN**          | Captures sequential dependencies in news content. |
| **LSTM**         | Handles long-term dependencies better than RNN. |
| **BERT**         | Transformer-based model with strong semantic understanding. |
| **Semantic Analysis** | Analyzes word meaning/context for better fake news identification. |

---

## 🪜 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Naive Bayes model**:
   ```bash
   python naive_bayes_model.py
   ```

4. **Run LSTM / RNN / BERT**:
   - You’ll find separate files:
     - `lstm_model.py`
     - `rnn_model.py`
     - `bert_classifier.py`

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📌 TODO

- [x] Data preprocessing & cleaning
- [x] Naive Bayes model
- [x] LSTM & RNN implementation
- [x] BERT classifier
- [x] Semantic analysis module
- [ ] Web UI integration
- [ ] Model deployment (e.g., Streamlit or Flask)

---

## 👥 Authors

- Team Members: Tanmay Khandelwal, Shubh Jain, Utkarsh Bansal

---
