# src/models.py
import nltk
import spacy
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.ner import NERecognizer

def train_ngram_model(sentences, n=2):
    """
    Trains a Laplace-smoothed N-gram model.
    Args:
        sentences: List of lists of tokens (e.g. [['hello', 'world'], ...])
        n: The 'n' in n-gram (default 2 for Bigram)
    Returns:
        model: Trained NLTK Laplace model
    """
    train_data, padded_sents = padded_everygram_pipeline(n, sentences)
    model = Laplace(n)
    model.fit(train_data, padded_sents)
    return model

class POSTagger:
    def __init__(self, language='english'):
        self.language = language
        if self.language == 'arabic':
            # Load CAMeL Tools MLE Disambiguator
            try:
                self.mle = MLEDisambiguator.pretrained('calima-msa-r13')
            except Exception:
                print("Error: CAMeL data not found. Run `camel_data -i calima-msa-r13`")

    def tag(self, tokens):
        if self.language == 'english':
            # NLTK POS Tagging
            return nltk.pos_tag(tokens)
        elif self.language == 'arabic':
            # CAMeL Tools Disambiguation
            disambig = self.mle.disambiguate(tokens)
            pos_tags = []
            for d in disambig:
                if d.analyses:
                    tag = d.analyses[0].analysis.get('pos', 'noun')
                else:
                    tag = 'noun'
                pos_tags.append((d.word, tag))
            return pos_tags

class NamedEntityRecognizer:
    def __init__(self, language='english'):
        self.language = language
        if self.language == 'english':
            # Load spaCy small model
            if not spacy.util.is_package("en_core_web_sm"):
                spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        elif self.language == 'arabic':
            # Load CAMeL Tools AraBERT NER
            try:
                self.ner = NERecognizer.pretrained('arabert')
            except Exception:
                print("Error: CAMeL NER data not found. Run `camel_data -i ner-arabert`")

    def predict(self, text_or_tokens):
        """
        Returns entities.
        English input: Raw string text.
        Arabic input: List of tokens.
        """
        if self.language == 'english':
            doc = self.nlp(text_or_tokens)
            return [(ent.text, ent.label_) for ent in doc.ents]
        elif self.language == 'arabic':
            labels = self.ner.predict_sentence(text_or_tokens)
            # Zip tokens with labels, filtering out 'O' (Outside)
            return [(tok, lbl) for tok, lbl in zip(text_or_tokens, labels) if lbl != 'O']