"""
Monkey patch for NLTK to fix broken punkt_tab resource lookup.
"""

import nltk
from nltk.data import load

def fixed_get_punkt_tokenizer(language):
    # Always load the standard punkt resource
    return load(f"tokenizers/punkt/{language}.pickle")

nltk.tokenize._get_punkt_tokenizer = fixed_get_punkt_tokenizer
