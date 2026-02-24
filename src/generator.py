# src/generator.py
from google import genai

def generate_resume_advice(
    resume_text: str, 
    jd_text: str, 
    overall_match_score: float, 
    matched_keywords_list: set,
    missing_keywords_list : set,
    keyword_coverage_percentage : float, 
    api_key: str
) -> str:
    """
    Constructs the prompt and calls the Gemini API to generate actionable advice.
    """
    # 1. Instantiate the client using the passed-in key
    client = genai.Client(api_key=api_key)

    # 2. Construct your massive f-string prompt here
    llm_prompt = f"""
    **System Persona:**
    You are an Expert Technical Recruiter and Senior Engineering Hiring Manager at a top-tier tech company. Your job is to analyze a candidate's resume against a specific Job Description (JD) and provide highly actionable, realistic advice to improve their ATS (Applicant Tracking System) performance and appeal to human reviewers.

    **Inputs & Calculated Metrics:**
    - Overall Match Score: {overall_match_score}
    - Keyword Coverage: {keyword_coverage_percentage}%
    - Matched Keywords: {matched_keywords_list}
    - Missing Keywords: {missing_keywords_list}
    - Resume Text: {resume_text}
    - Job Description: {jd_text}

    **Objective:**
    Review the provided Resume, Job Description, and Calculated Metrics. Identify areas where the candidate's existing experience aligns with the JD but is currently understated. Suggest targeted rewrites for 3-4 specific bullet points to naturally incorporate the missing keywords and improve the overall match score.

    **Strict Guardrails & Rules:**
    1. **NO Keyword Stuffing:** Do not artificially force exact JD keywords into sentences where they do not naturally belong. Focus on *semantic alignment*. If the JD asks for "scalable systems," highlighting an accomplishment about processing "large-scale datasets" is sufficient.
    2. **NO Meta-Commentary:** Resume bullets must be in the format of Action Verb + Context + Result (e.g., "Engineered X using Y, resulting in Z"). Do NOT add phrases like "Demonstrated ability to...".
    3. **NO Hallucinations:** You may only use facts, metrics, and tools explicitly stated in the provided resume. Do not invent experience, seniority, or skills.
    4. **Prioritize Hard Metrics:** If the original bullet has a metric (e.g., "reduced latency", "94% accuracy"), the rewritten bullet MUST retain that metric.

    **Required Output Format:**
    **1. Scorecard:** Present the Overall Match Score and Keyword Coverage as a clean, simple summary.
    **2. Thematic Alignment Assessment:** Briefly explain (2-3 sentences) how the candidate's actual experience matches the core problems the engineering team is trying to solve.
    **3. Strategic Keyword Gaps:** Identify the most critical `Missing Keywords` from the inputs and explain *why* they matter for this specific role.
    **4. High-Impact Rewrites:** Provide 3-4 suggested bullet point rewrites. For each, show the:
    * *Original Bullet:*
    * *Why it needs changing:* (Focus on how integrating specific missing keywords or JD themes improves the bullet).
    * *Revised Bullet:* (Strictly adhering to the Action + Context + Result format).
    """

    # 3. Call the model and return the text
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=llm_prompt
    )
    
    return response.text