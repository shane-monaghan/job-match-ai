from sentence_transformers import SentenceTransformer, util
import numpy as np

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

if __name__ == "__main__":
    model = load_model('all-MiniLM-L6-v2')

    resume_text = open('data/resume_text.txt', encoding='utf-8').read()
    job_description_text = open('data/job_description_text.txt', encoding='utf-8').read()

    similarity_score = calculate_similarity_score(model, resume_text, job_description_text)
    print(similarity_score)