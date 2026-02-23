import streamlit as st
from src.parser import extract_text
from src.processor import load_model, calculate_similarity_score, get_stop_words, get_keywords
from src.generator import generate_resume_advice

# 1. Page Setup
st.set_page_config(page_title="Career Match AI", page_icon="🚀")

st.title("Career Match AI")
st.markdown("""
    Upload your resume and paste a job description to see how well they align.
""")

# 2. Cached Model Loading
# This ensures the model only loads ONCE when the app starts
@st.cache_resource
def get_model():
    return load_model('all-MiniLM-L6-v2')

model = get_model()
stop_words = get_stop_words()

# 3. Input UI
st.divider()

# File Uploader for the PDF
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Text Area for Job Description
job_description = st.text_area("Paste Job Description", height=300, placeholder="Enter the full job posting text here...")

# 4. Execution Logic
if st.button("Calculate Match Score", type="primary"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Analyzing..."):
            try:
                # Use your src/parser logic
                resume_text = extract_text(uploaded_file)
                
                # Use your src/processor logic
                cosine_score = calculate_similarity_score(model, resume_text, job_description)

                resume_keywords = get_keywords(resume_text, stop_words, 2)
                jd_keywords = get_keywords(job_description, stop_words, 2)

                matched_keywords = resume_keywords & jd_keywords
                missing_keywords = jd_keywords - resume_keywords

                keyword_matching_score = len(matched_keywords) / len(jd_keywords)

                # Display Result
                st.success("Analysis Complete!")
                st.metric(label="Similarity Score", value=f"{cosine_score:.2f}")
                
                st.metric(label="Keyword Coverage", value=f"{keyword_matching_score:.2%}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### ✅ Matched")
                    st.write(", ".join(list(matched_keywords)[:15])) # Show top 15

                with col2:
                    st.write("### ❌ Missing")
                    st.write(", ".join(list(missing_keywords)[:15]))
                                
                # Bonus: Visual feedback
                if cosine_score > 0.7:
                    st.balloons()

                # Generate Advice
                api_key = st.secrets["GEMINI_API_KEY"]
                
                advice = generate_resume_advice(
                    resume_text=resume_text,
                    job_description=job_description,
                    cosine_score=cosine_score,
                    missing_keywords=missing_keywords,
                    api_key=api_key
                )
                
                # Display Result
                st.write("### 🤖 Career Coach Advice")
                st.write(advice)
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please provide both a resume PDF and a job description.")