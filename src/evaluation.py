# src/evaluation.py
from nltk.util import ngrams

def calculate_perplexity(model, test_sentences, n=2):
    """
    Calculates perplexity of an N-gram model on test data.
    Args:
        model: Trained NLTK LM.
        test_sentences: List of lists of tokens.
        n: N-gram order.
    """
    test_bigrams = [
        ngrams(sent, n, pad_left=True, pad_right=True, 
               left_pad_symbol='<s>', right_pad_symbol='</s>') 
        for sent in test_sentences
    ]
    
    # Flatten list
    test_data = [gram for sent in test_bigrams for gram in sent]
    
    # Filter None values just in case
    test_data_filtered = [gram for gram in test_data if None not in gram]
    
    perplexity = model.perplexity(test_data_filtered)
    return perplexity