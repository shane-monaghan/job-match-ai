from sentence_transformers import util
from typing import Any

def calculate_similarity_score(
    resume_embedding: Any, 
    jd_embedding: Any
) -> float:
    """Calculates the cosine similarity score between a resume and a job description.

    This function takes pre-computed vector embeddings for a resume and a job
    description and computes their cosine similarity to determine how closely 
    the candidate's profile matches the role requirements.

    Args:
        resume_embedding (Any): The pre-computed vector embedding representing 
            the candidate's resume (typically a PyTorch Tensor or NumPy array).
        jd_embedding (Any): The pre-computed vector embedding representing 
            the job description (typically a PyTorch Tensor or NumPy array).

    Returns:
        float: A cosine similarity score representing the semantic match. Higher 
            values indicate a stronger match (typically ranging from -1.0 to 1.0, 
            or 0.0 to 1.0 depending on the underlying embedding model).
    """
    # Calculate the cosine of the angle between the two embeddings
    cosine_scores = util.cos_sim(resume_embedding, jd_embedding)

    # .item() extracts the standard Python float from the PyTorch tensor
    return cosine_scores[0][0].item()

def calculate_section_similarities(
    resume_vectors : dict[str, list[Any]],
    jd_vector : Any
) -> dict[str, float]:
    """Calculates the highest similarity score for each resume section against a job description.

    This function iterates through different sections of a resume (e.g., "Experience",
    "Skills"), where each section is represented by a list of vector embeddings 
    (chunks). It compares each chunk to the provided job description vector and 
    records the maximum similarity score achieved for that specific section.

    Args:
        resume_vectors (dict[str, list[Any]]): A dictionary mapping resume section 
            names to lists of their respective chunk embeddings.
        jd_vector (Any): The pre-computed vector embedding representing the full 
            job description or a specific job requirement.

    Returns:
        dict[str, float]: A dictionary mapping each resume section name to its 
            highest calculated cosine similarity score.
    """
    section_scores = {}

    for section, chunk_vectors in resume_vectors.items():
        best_score = 0

        for chunk_vec in chunk_vectors:
            chunk_score = calculate_similarity_score(chunk_vec, jd_vector)

            best_score = chunk_score if chunk_score > best_score else best_score

        section_scores[section] = best_score

    return section_scores