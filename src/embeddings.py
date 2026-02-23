from sentence_transformers import SentenceTransformer
from torch import Tensor

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

def get_embedding(text : str, model : SentenceTransformer) -> Tensor:
    """
    Generates an embedding for the given text using a SentenceTransformer model.

    Args:
        text (str): The input string to be encoded.
        model (SentenceTransformer): The pre-trained SentenceTransformer model 
            used to generate the embedding.

    Returns:
        Tensor: A PyTorch Tensor representing the embedded text.
    """
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding
