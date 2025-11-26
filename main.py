# run_pipeline.py
import json
import os
import sys
from collections import Counter

# Add src/ directory to the path to import modules
sys.path.append(os.path.abspath('src'))

# Import the modular functions
from src.preprocessing import clean_text, tokenize, get_chunk_parser
from src.models import train_ngram_model, POSTagger, NamedEntityRecognizer
from src.evaluation import calculate_perplexity

# --- Configuration ---
N_GRAM_ORDER = 2  # Bigrams
TRAIN_SPLIT = 0.8 # 80% for training LM

def load_raw_data():
    """Load English and Arabic articles from JSON files."""
    try:
        with open('data/bbc-english-news.json', 'r', encoding='utf-8') as f:
            english_data = json.load(f)
        english_articles = [article['text'] for article in english_data]
    except FileNotFoundError:
        print("Error: English data file not found in data/.")
        english_articles = []

    try:
        with open('data/arabic-news.json', 'r', encoding='utf-8') as f:
            arabic_data = json.load(f)
        arabic_articles = [article['text'] for article in arabic_data]
    except FileNotFoundError:
        print("Error: Arabic data file not found in data/.")
        arabic_articles = []
        
    return english_articles, arabic_articles

def run_language_pipeline(articles, language):
    print(f"\n{'='*50}\nRUNNING {language.upper()} PIPELINE ({len(articles)} ARTICLES)\n{'='*50}")

    # --- 1. Preprocessing & Tokenization ---
    print("1. Preprocessing & Tokenization...")
    all_sentences = []
    
    # Sentence tokenization and cleaning must happen per article
    for article in articles:
        # Use NLTK's sentence tokenizer (assuming appropriate setup)
        sentences = nltk.sent_tokenize(article)
        for sent in sentences:
            cleaned_text = clean_text(sent, language)
            if cleaned_text:
                all_sentences.append(tokenize(cleaned_text, language))
    
    # Split data for LM training/testing
    split_point = int(len(all_sentences) * TRAIN_SPLIT)
    train_sents = all_sentences[:split_point]
    test_sents = all_sentences[split_point:]

    # Flatten lists for statistics
    all_tokens = [token for sent in all_sentences for token in sent]
    
    print(f"   Total tokens: {len(all_tokens):,}")
    print(f"   Unique tokens: {len(set(all_tokens)):,}")
    print(f"   Training sentences: {len(train_sents)}")

    # --- 2. N-gram Language Modeling (LM) & Evaluation ---
    print(f"\n2. Training N={N_GRAM_ORDER} Model...")
    try:
        lm_model = train_ngram_model(train_sents, N_GRAM_ORDER)
        perplexity = calculate_perplexity(lm_model, test_sents, N_GRAM_ORDER)
        print(f"   ✅ Perplexity on Test Set: {perplexity:.2f}")
    except Exception as e:
        print(f"   ❌ Error during LM/Perplexity calculation: {e}")


    # --- 3. POS Tagging & Chunking ---
    print("\n3. POS Tagging and Chunking...")
    
    # Initialize Tagger and Chunk Parser
    pos_tagger = POSTagger(language)
    chunk_parser = get_chunk_parser(language)
    
    # Use first test sentence for demonstration
    sample_tokens = test_sents[0]
    pos_tags = pos_tagger.tag(sample_tokens)
    
    # Perform Chunking
    chunk_tree = chunk_parser.parse(pos_tags)
    
    # Extract chunks
    chunks = [
        " ".join([word for word, pos in subtree.leaves()]) 
        for subtree in chunk_tree.subtrees() if subtree.label() == 'NP'
    ]
    
    print(f"   Sample Tokens: {sample_tokens[:10]}...")
    print(f"   Sample POS Tags (First 5): {pos_tags[:5]}...")
    print(f"   Sample Chunks Extracted: {chunks[:3]}...")


    # --- 4. Named Entity Recognition (NER) ---
    print("\n4. Named Entity Recognition (NER)...")
    
    ner_processor = NamedEntityRecognizer(language)
    all_entities = []

    # Iterate through all articles for statistics
    for article in articles:
        # NER typically runs better on the full text (especially spaCy/AraBERT)
        if language == 'english':
            entities = ner_processor.predict(article)
            all_entities.extend([label for text, label in entities])
        elif language == 'arabic':
            # Needs tokenized input for CAMeL Tools NER
            tokens_for_ner = tokenize(clean_text(article, language), language)
            entities = ner_processor.predict(tokens_for_ner)
            all_entities.extend([label for text, label in entities])


    entity_counts = Counter(all_entities)
    print(f"   Total Entities Found: {len(all_entities):,}")
    print(f"   Top 5 Entity Types: {entity_counts.most_common(5)}")

    
    print(f"\n{language.upper()} PIPELINE FINISHED.\n{'='*50}")


if __name__ == "__main__":
    # Ensure NLTK and other resources are downloaded/configured before execution
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK data missing. Please run `./setup.sh` first!")
        sys.exit(1)

    english_articles, arabic_articles = load_raw_data()
    
    if english_articles:
        run_language_pipeline(english_articles, 'english')
    
    if arabic_articles:
        run_language_pipeline(arabic_articles, 'arabic')