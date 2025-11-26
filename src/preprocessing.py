# src/preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from camel_tools.tokenizers.word import simple_word_tokenize

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text, language='english'):
    """
    Cleans text by removing URLs, punctuation, numbers, and normalizing.
    """
    if text is None:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if language == 'english':
        text = text.lower()
    elif language == 'arabic':
        # Arabic-specific normalization (from your notebook)
        text = re.sub("[إأآ]", "ا", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("ـ", "", text) # Remove tatweel

    return text

def tokenize(text, language='english'):
    """
    Tokenizes text based on language.
    English: NLTK word_tokenize
    Arabic: CAMeL Tools simple_word_tokenize
    """
    if language == 'english':
        return nltk.word_tokenize(text)
    elif language == 'arabic':
        return simple_word_tokenize(text)
    else:
        raise ValueError("Language must be 'english' or 'arabic'")

def get_chunk_parser(language='english'):
    """
    Returns a configured NLTK RegexpParser based on the grammar defined in the notebook.
    """
    if language == 'english':
        # Grammar from Task 4
        grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    elif language == 'arabic':
        # Grammar from Task 4 (CAMeL tags)
        grammar = """
        NP: {<noun.*>+<adj.*>*}
            {<digit>+}
        """
    else:
        raise ValueError("Language must be 'english' or 'arabic'")
        
    return nltk.RegexpParser(grammar)