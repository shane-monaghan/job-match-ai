# src/generator.py
from google import genai

def generate_resume_advice(
    resume_text: str, 
    job_description: str, 
    cosine_scores: float, 
    missing_keywords: set, 
    api_key: str
) -> str:
    """
    Constructs the prompt and calls the Gemini API to generate actionable advice.
    """
    # 1. Instantiate the client using the passed-in key
    client = genai.Client(api_key=api_key)

    # 2. Construct your massive f-string prompt here
    prompt = f"""
        You are an expert career coach. 

        Data:
        - Cosine Similarity Scores per Resume Section: {cosine_scores}
        - Missing Keywords: {missing_keywords}

        Context:
        - User Resume: {resume_text}
        - Job Description: {job_description}

        Task:
        Based on the missing keywords and the cosine similarities per section, identify weakpoints in the resume that can be improved.
        Provide feedback by showing the original line and then your suggested rewritten line.
        Ensure that any rewrites are grounded in user's resume. Do not write line improvements that may make up accomplishments the user does not have.
        Ensure improvements are made with the goal of improving the resume's alignment with the job description.
        At the end, provide a short paragraph assessing the user's competitiveness for the job.
    """

    # 3. Call the model and return the text
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text