@echo off
REM --- Windows Setup Script for Arabic-English NLP Pipeline ---
REM This script downloads necessary data for NLTK, spaCy, and CAMeL Tools.

echo.
echo [1/3] Downloading NLTK and spaCy Data...
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader averaged_perceptron_tagger
python -m spacy download en_core_web_sm

echo.
echo [2/3] Downloading CAMeL Tools Data (Morphology)...
REM Ensure the 'light' package, which includes the MLE Disambiguator, is installed.
camel_data -i light

echo.
echo [3/3] Downloading CAMeL Tools NER Model (AraBERT)...
camel_data -i ner-arabert

echo.
echo --- Setup Complete! ---
echo.
pause