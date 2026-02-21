import streamlit as st
from src.parser import extract_text
from src.processor import load_model, calculate_similarity_score

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
                score = calculate_similarity_score(model, resume_text, job_description)
                
                # Display Result
                st.success("Analysis Complete!")
                st.metric(label="Similarity Score", value=f"{score:.2f}")
                
                # Bonus: Visual feedback
                if score > 0.7:
                    st.balloons()
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please provide both a resume PDF and a job description.")