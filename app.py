import streamlit as st
from src.parser import extract_text, chunk_resume
from src.processor import load_model, calculate_section_similarities, get_stop_words, get_keywords
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
                resume_sections = chunk_resume(resume_text)
                similarity_scores = calculate_section_similarities(model, resume_sections, job_description)

                st.success("Analysis Complete!")

                # 1. Calculate an aggregate score dynamically from whatever sections were found
                # We filter out CONTACT INFO so it doesn't skew the average
                meaningful_scores = [score for section, score in similarity_scores.items() if section != "CONTACT INFO"]
                overall_score = sum(meaningful_scores) / len(meaningful_scores) if meaningful_scores else 0

                # 2. Display Overall Metric
                st.metric(label="Overall Match Quality", value=f"{overall_score:.2f}")
                st.progress(min(max(overall_score, 0.0), 1.0)) # Ensure it's between 0 and 1

                st.write("### 📑 Section Scorecard")
                st.caption("Similarity scores for each identified section of your resume:")

                # 3. Dynamic Grid for Dictionary Sections
                # We use 3 columns for a balanced look
                cols = st.columns(3)
                current_col = 0

                for section, score in similarity_scores.items():
                    # Skip contact info as it's not a performance metric
                    if section == "CONTACT INFO":
                        continue
                    
                    with cols[current_col % 3]:
                        # Visual color coding based on similarity
                        if score > 0.75:
                            sentiment = "Inverse" # Streamlit's way to highlight
                            label_prefix = "✅"
                        elif score > 0.5:
                            label_prefix = "⚠️"
                        else:
                            label_prefix = "❌"
                            
                        st.metric(label=f"{label_prefix} {section}", value=f"{score:.2f}")
                        
                    current_col += 1

                st.divider()

                resume_keywords = get_keywords(resume_text, stop_words, 2)
                jd_keywords = get_keywords(job_description, stop_words, 2)

                matched_keywords, missing_keywords, keyword_matching_score = calculate_keyword_coverage(resume_keywords, jd_keywords)
                
                st.metric(label="Keyword Coverage", value=f"{keyword_matching_score:.2%}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### ✅ Matched")
                    st.write(", ".join(list(matched_keywords)[:15])) # Show top 15

                with col2:
                    st.write("### ❌ Missing")
                    st.write(", ".join(list(missing_keywords)[:15]))
                                
                # Generate Advice
                api_key = st.secrets["GEMINI_API_KEY"]
                
                advice = generate_resume_advice(
                    resume_text=resume_text,
                    job_description=job_description,
                    cosine_scores=similarity_scores,
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