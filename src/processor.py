from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('stopwords')
nltk.download('punkt') # Required for tokenization
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng') # The engine that identifies grammar
nltk.download('universal_tagset') # Simplifies tags to 'NOUN', 'ADJ', etc.

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def calculate_similarity_score(
    model: SentenceTransformer,
    resume_text: str, 
    job_description: str
) -> float:
    """
    Calculates the semantic similarity between a resume and a job description.

    This function transforms raw text into vector embeddings using the provided
    SentenceTransformer model and computes the cosine similarity to determine 
    how closely the candidate's profile matches the role requirements.

    Args:
        model (SentenceTransformer): The pre-trained AI model used for encoding.
        resume_text (str): The cleaned text extracted from a candidate's resume.
        job_description (str): The text content of the target job posting.

    Returns:
        float: A similarity score, typically between 0.0 and 1.0, where 
            higher values indicate a stronger semantic match.
    """
    # Generate high-dimensional vector representations of the texts
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    # Calculate the cosine of the angle between the two vectors
    cosine_scores = util.cos_sim(resume_embedding, job_embedding)

    # .item() extracts the standard Python float from the PyTorch tensor
    return cosine_scores[0][0].item()

def load_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Initializes and loads a pre-trained SentenceTransformer model.

    This function fetches the specified model weights (from local cache or 
    the Hugging Face Hub) and loads them into memory for generating 
    text embeddings.

    Args:
        model_name (str): The name or path of the transformer model to load. 
            Defaults to 'all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer: An instance of the SentenceTransformer class 
            ready for encoding text.
    """
    return SentenceTransformer(model_name)

def remove_stop_words(words : list[str], stop_words : list[str]) -> set[str]:
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    return set(filtered_words)

def get_keywords(words: list[str], stop_words: list[str], min_length: int) -> set[str]:
    """
    Refines a word list by applying lowercasing, stop-word removal, 
    alpha-only filtering, length constraints, and POS tagging in one pass.
    """
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

if __name__ == "__main__":
    model = load_model('all-MiniLM-L6-v2')

    resume_text = open('data/resume_text.txt', encoding='utf-8').read()
    jd_text = open('data/job_description_text.txt', encoding='utf-8').read()

    similarity_score = calculate_similarity_score(model, resume_text, jd_text)
    print(similarity_score)

    resume_words = word_tokenize(resume_text)
    jd_words = word_tokenize(jd_text)

    stop_words = set(stopwords.words('english'))

    resume_keywords = get_keywords(resume_words, stop_words, 2)
    jd_keywords = get_keywords(jd_words, stop_words, 2)
    missing_keywords = jd_keywords - resume_keywords

    print(missing_keywords)