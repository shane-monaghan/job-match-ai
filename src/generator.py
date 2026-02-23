# src/generator.py
from google import genai

def generate_resume_advice(
    resume_text: str, 
    job_description: str, 
    cosine_score: float, 
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
        - Cosine Similarity Score: {cosine_score:.2f}
        - Missing Keywords: {missing_keywords}

        Context:
        - User Resume: {resume_text}
        - Job Description: {job_description}

        Task:
        Based on the missing keywords, identify exactly which bullet points in the user's resume should be rewritten. Provide 3 specific, actionable examples. In 3-4 sentences, explain what they could do better in their resume to be more competitive for this job.
    """

    # 3. Call the model and return the text
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    return response.text