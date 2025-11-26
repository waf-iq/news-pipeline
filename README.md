# Arabic-English News NLP Pipeline

## 📌 Project Overview
This project engineers a complete Natural Language Processing (NLP) pipeline for bilingual news text (Arabic and English). It processes news corpora to extract linguistic insights using a combination of rule-based and statistical methods.

The pipeline focuses on comparing NLP techniques across two morphologically distinct languages, highlighting the challenges of Arabic text processing.

## 🚀 Features
The pipeline implements the following stages for both languages:
1.  **Data Ingestion & Cleaning:** normalization and noise removal (URLs, punctuation).
2.  **Tokenization:** NLTK for English; CAMeL Tools `simple_word_tokenize` for Arabic.
3.  **Language Modeling:** N-gram models (Bigrams) with Laplace smoothing and Perplexity evaluation.
4.  **POS Tagging:** NLTK `averaged_perceptron_tagger` (English) and CAMeL Tools MLE Disambiguator (Arabic).
5.  **Shallow Parsing (Chunking):** Rule-based `RegexpParser` for Noun Phrase (NP) extraction.
6.  **Named Entity Recognition (NER):** spaCy (English) and CAMeL Tools/AraBERT (Arabic).

## 📂 Dataset
The project uses a dataset of ~1,000-2,000 sentences balanced across:
* **English:** BBC News articles (`bbc-english-news.json`)
* **Arabic:** Collected Arabic news articles (`arabic-news.json`)

*Note: All data is anonymized and collected from publicly available sources in compliance with robots.txt.*

## 🛠️ Tech Stack
* **Language:** Python 3.x 
* **Core Libraries:** NLTK, Pandas, NumPy
* **Arabic NLP:** CAMeL Tools (Morphology, NER, Tokenization) 
* **English NLP:** spaCy
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Installation & Setup

To reproduce the results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/waf-iq/news-pipeline.git
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


## 📜 License
MIT
