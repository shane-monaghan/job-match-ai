import streamlit as st
from src import ingestion, chunking, nlp_extraction, embeddings, scoring
from src.generator import generate_resume_advice

# 1. Page Setup
st.set_page_config(page_title="Career Match AI", page_icon="🚀")

st.title("Career Match AI")
st.markdown("""
    Upload your resume and paste a job description to see how well they align.
""")

# 2. Cached Initialization & Model Loading
# This sets up NLTK and the AI model ONCE when the app starts
nlp_extraction.setup_nltk()

@st.cache_resource
def get_model():
    return embeddings.load_model('all-MiniLM-L6-v2')

model = get_model()
stop_words = nlp_extraction.get_stop_words()

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
                # ==========================================
                # THE ORCHESTRATION PIPELINE
                # ==========================================
                
                # 1. Ingestion Layer
                resume_text = ingestion.extract_text(uploaded_file)
                
                # 2. Chunking Layer
                resume_sections = chunking.chunk_resume(resume_text)
                
                # 3. NLP Extraction Layer
                resume_keywords = nlp_extraction.get_keywords(resume_text, stop_words, 2)
                jd_keywords = nlp_extraction.get_keywords(job_description, stop_words, 2)
                
                # 4. Embedding Layer (Heavy ML)
                jd_vector = embeddings.get_embedding(job_description, model)
                
                resume_vectors_dict = {}
                for section, chunks in resume_sections.items():
                    # Convert each text chunk in the section into a tensor
                    resume_vectors_dict[section] = [
                        embeddings.get_embedding(chunk, model) for chunk in chunks
                    ]
                
                # 5. Scoring Layer (Pure Math)
                similarity_scores = scoring.calculate_section_similarities(resume_vectors_dict, jd_vector)
                matched_keywords, missing_keywords, keyword_matching_score = scoring.calculate_keyword_coverage(resume_keywords, jd_keywords)


                # ==========================================
                # PRESENTATION LAYER (UI)
                # ==========================================

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
                    jd_text=job_description,
                    overall_match_score=overall_score,
                    matched_keywords_list=matched_keywords,
                    missing_keywords_list=missing_keywords,
                    keyword_coverage_percentage=keyword_matching_score,
                    api_key=api_key
                )
                
                # Display Result
                st.write("### 🤖 Career Coach Advice")
                st.write(advice)
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please provide both a resume PDF and a job description.")