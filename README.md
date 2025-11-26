# Arabic-English News NLP Pipeline

## 📌 Project Overview
This project engineers a complete Natural Language Processing (NLP) pipeline for bilingual news text (Arabic and English). [cite_start]It processes news corpora to extract linguistic insights using a combination of rule-based and statistical methods.

[cite_start]The pipeline focuses on comparing NLP techniques across two morphologically distinct languages, highlighting the challenges of Arabic text processing[cite: 30].

## 🚀 Features
The pipeline implements the following stages for both languages:
1.  **Data Ingestion & Cleaning:** normalization and noise removal (URLs, punctuation).
2.  **Tokenization:** NLTK for English; CAMeL Tools `simple_word_tokenize` for Arabic.
3.  [cite_start]**Language Modeling:** N-gram models (Bigrams) with Laplace smoothing and Perplexity evaluation[cite: 26].
4.  [cite_start]**POS Tagging:** NLTK `averaged_perceptron_tagger` (English) and CAMeL Tools MLE Disambiguator (Arabic)[cite: 27].
5.  [cite_start]**Shallow Parsing (Chunking):** Rule-based `RegexpParser` for Noun Phrase (NP) extraction[cite: 28].
6.  [cite_start]**Named Entity Recognition (NER):** spaCy (English) and CAMeL Tools/AraBERT (Arabic)[cite: 29].

## 📂 Dataset
[cite_start]The project uses a dataset of ~1,000-2,000 sentences [cite: 35] balanced across:
* **English:** BBC News articles (`bbc-english-news.json`)
* **Arabic:** Collected Arabic news articles (`arabic-news.json`)

[cite_start]*Note: All data is anonymized and collected from publicly available sources in compliance with robots.txt[cite: 36, 37].*

## 🛠️ Tech Stack
* [cite_start]**Language:** Python 3.x [cite: 42]
* **Core Libraries:** NLTK, Pandas, NumPy
* [cite_start]**Arabic NLP:** CAMeL Tools (Morphology, NER, Tokenization) [cite: 43]
* **English NLP:** spaCy
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Installation & Setup

[cite_start]To reproduce the results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/Arabic-English-NLP-Pipeline.git](https://github.com/yourusername/Arabic-English-NLP-Pipeline.git)
    cd Arabic-English-NLP-Pipeline
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Setup Script:**
    This script downloads necessary NLTK data, spaCy models, and CAMeL Tools databases.
    ```bash
    bash setup.sh
    ```

## 📊 Evaluation & Results
* **Language Models:** Perplexity scores are calculated for both languages to evaluate the bigram models on a held-out test set (20% split).
* **NER:** Precision/Recall analysis of entity extraction.
* **Morphology:** Discussion on Arabic clitic handling and normalization is included in the final report.

## 👥 Team
* **Project Lead:** [Name]
* **Data & Pre-processing Lead:** [Name]
* **Modeling Lead:** [Name]
* **Evaluation Lead:** [Name]
* **Documentation Lead:** [Name]

## 📜 License
MIT
