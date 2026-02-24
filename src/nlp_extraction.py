import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@st.cache_resource
def setup_nltk():
    """
    Downloads required NLTK datasets. 
    Using @st.cache_resource ensures this executes exactly ONCE 
    per app session, drastically speeding up reload times.
    """
    # quiet=True stops it from printing "Downloading package..." to the console every time
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    
    return True # Streamlit cache functions need to return something

def remove_stop_words(words : list[str], stop_words : list[str]) -> set[str]:
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    return set(filtered_words)

@st.cache_data
def get_stop_words() -> set[str]:
    stop_words = set(stopwords.words('english'))
    return stop_words

def get_keywords(text: str, stop_words: list[str], min_length: int) -> set[str]:
    """
    Refines a word list by applying lowercasing, stop-word removal, 
    alpha-only filtering, length constraints, and POS tagging in one pass.
    """
    # Tokenize words
    words = word_tokenize(text)
    
    # Standardize stop_words for faster lookups
    stop_words_set = {sw.lower() for sw in stop_words}
    
    # Pre-process POS tags to avoid redundant calls inside a loop
    tagged_words = nltk.pos_tag(words, tagset='universal', lang='eng')
    
    return {
        word.lower() for word, pos in tagged_words
        if word.isalpha() 
        and word.lower() not in stop_words_set
        and len(word) > min_length
        and pos in {'NOUN', 'ADJ'}
    }