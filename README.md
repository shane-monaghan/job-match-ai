# Career Match AI

An intelligent resume-to-job-description matching system powered by NLP embeddings and semantic analysis. Analyzes how well your resume aligns with job postings and provides actionable recommendations for improvement.

## 📋 Overview

**Career Match AI** is a web-based application built with Streamlit that leverages state-of-the-art NLP techniques to evaluate resume-job description alignment. Instead of simple keyword matching, the system uses semantic embeddings to understand the meaning and context of both documents, providing a more accurate and nuanced matching score.

The application generates multiple scoring metrics and uses AI to produce targeted, actionable advice for improving your resume's alignment with specific job postings.

## 🎯 Goals

1. **Accurate Semantic Matching**: Move beyond keyword matching to understand meaningful alignment between resume skills/experience and job requirements.

2. **Actionable Insights**: Provide specific, implementable recommendations for resume improvements tailored to each job posting.

3. **Multi-Faceted Scoring**: Evaluate matching at multiple levels—semantic similarity across sections, keyword coverage, and contextual relevance.

4. **User-Friendly Interface**: Make advanced NLP analysis accessible to job seekers through an intuitive web interface.

5. **ATS Optimization**: Help candidates optimize their resumes for both Applicant Tracking Systems (ATS) and human recruiters.

## 🔧 Methodology

The system employs a modular pipeline architecture with six distinct processing stages:

### 1. **Ingestion & Text Extraction** (`ingestion.py`)
- Extracts raw text from PDF resumes using the `pdfplumber` library
- Normalizes Unicode characters and removes PDF artifacts
- Cleans output to ASCII-safe format for downstream processing
- Handles multi-page documents seamlessly

### 2. **Structural Chunking** (`chunking.py`)
- Analyzes resume text to identify standard sections (Experience, Skills, Education, etc.)
- Splits each section into manageable chunks with overlapping windows
- Preserves context by maintaining section headers through chunks
- Enables section-by-section similarity scoring

**Parameters:**
- Configurable max chunk size (token count)
- Overlap windows maintain semantic continuity between chunks

### 3. **NLP Keyword Extraction** (`nlp_extraction.py`)
- Uses NLTK tokenization and POS (Part-of-Speech) tagging to extract meaningful keywords
- Filters by word type (nouns and adjectives only) to capture domain-relevant terms
- Removes stopwords (common English words like "the", "and", "is")
- Applies minimum length constraints to exclude single-letter artifacts

**Algorithm:**
```
For each word in text:
  1. Tokenize and lowercase
  2. Apply POS tagging
  3. Keep only NOUN and ADJ
  4. Remove stopwords
  5. Enforce minimum length constraint
```

### 4. **Semantic Embeddings** (`embeddings.py`)
- Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model
- Converts text chunks and job descriptions into 384-dimensional dense vectors
- These embeddings capture semantic meaning in a way that keyword matching cannot
- Model is cached for performance optimization

**Vector Representation:**
- Each chunk and the full job description are encoded into a shared semantic space
- Mathematically similar texts (even with different words) will have similar vectors

### 5. **Similarity Scoring** (`scoring.py`)
- **Semantic Similarity**: Calculates cosine similarity between job description vector and each resume section chunk
- **Keyword Coverage**: Computes the percentage of job description keywords present in the resume
- **Section Scores**: Aggregates the best-matching chunk per section to create a section-level score
- **Overall Score**: Averages meaningful section scores (excluding contact info) into a 0-1 similarity metric

**Scoring Equations:**
- Cosine Similarity: $\text{similarity} = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}$
- Keyword Coverage: $\text{coverage} = \frac{|\text{matched keywords}|}{|\text{total JD keywords}|}$
- Overall Match: $\text{score} = \frac{\sum \text{section scores}}{n \text{ sections}}$

### 6. **AI-Powered Recommendations** (`generator.py`)
- Uses Google's Gemini API to generate expert-level resume improvement advice
- Analyzes gaps between resume and job requirements
- Provides 3-4 targeted bullet point rewrites
- Ensures recommendations are realistic, avoid keyword stuffing, and maintain factual accuracy

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│          (app.py - User interaction & result visualization) │
└──────────────┬──────────────────────────────────────────────┘
               │
     ┌─────────▼──────────┐
     │   INGESTION LAYER  │
     │   (PDF → Text)     │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────┐
     │   CHUNKING LAYER   │
     │  (Text → Chunks)   │
     └─────────┬──────────┘
               │
     ┌─────────┴─────────────────┐
     │                           │
 ┌───▼────────────┐    ┌────────▼──────┐
 │ NLP EXTRACTION │    │  EMBEDDINGS    │
 │ (Keywords)     │    │  (Vectors)     │
 └───┬────────────┘    └────────┬──────┘
     │                          │
     └──────────┬───────────────┘
                │
         ┌──────▼──────────┐
         │  SCORING LAYER  │
         │  (Similarity &  │
         │  Coverage)      │
         └──────┬──────────┘
                │
         ┌──────▼──────────┐
         │  GENERATOR      │
         │  (Gemini API)   │
         └─────────────────┘
```

## 💻 Installation

### Prerequisites
- Python 3.8+
- Pip package manager
- Google Gemini API key (for recommendations feature)

### Setup Instructions

1. **Navigate to the project directory:**
   ```bash
   cd job-match-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - Upload a PDF resume
   - Paste a job description
   - Click "Calculate Match Score"

## 📊 Key Libraries & Models

| Component | Library | Purpose |
|-----------|---------|---------|
| Web Framework | Streamlit | Interactive UI and caching |
| PDF Processing | pdfplumber | Resume text extraction |
| Embeddings | SentenceTransformers | Semantic vector encoding |
| NLP Tasks | NLTK | Tokenization, POS tagging, stopword removal |
| Similarity | PyTorch + scipy | Cosine similarity calculations |
| AI Recommendations | Google Gemini | LLM-based advice generation |
| Data Processing | Pandas, NumPy | General data manipulation |

## 📈 Output Metrics

The application provides multiple levels of analysis:

1. **Overall Match Score** (0.0 - 1.0)
   - Weighted average of all resume section similarities
   - Higher is better; indicates how well overall profile matches

2. **Section Scorecard**
   - Individual scores for Experience, Skills, Education, etc.
   - Visual color coding (green/yellow/red) for quick assessment
   - Identifies strongest and weakest resume sections

3. **Keyword Analysis**
   - Count of matched vs. missing keywords
   - Percentage coverage (matched / total JD keywords)
   - Specific lists of found and missing critical terms

4. **AI Recommendations**
   - Thematic alignment assessment
   - Strategic gap analysis
   - 3-4 targeted bullet point rewrites with explanations

## ⚠️ Limitations

### 1. **Model Limitations**
- **Embedding Model Constraints**: The `all-MiniLM-L6-v2` model is optimized for semantic similarity but has:
  - Maximum input length of ~512 tokens (≈400 words)
  - 384-dimensional vector space (lower resolution than larger models)
  - Trained on general English text, not specialized technical jargon
- **Domain Specificity**: Embeddings may not perfectly understand niche industry terminology or emerging tech stacks

### 2. **Resume Parsing Limitations**
- **Structural Assumptions**: The chunking system relies on detecting standard resume headers (Experience, Skills, etc.)
  - Non-standard resume formats may not parse correctly
  - Sections in unexpected orders or with non-standard names may be misclassified
- **PDF Extraction Issues**: 
  - Image-based PDFs (scanned documents) cannot be processed
  - Complex PDF layouts with multiple columns may extract incorrectly
  - Formatting and spacing information is lost

### 3. **Semantic Matching Constraints**
- **Context Loss**: Chunk-level analysis may miss cross-section dependencies
  - A skill mentioned in "Experience" won't directly connect to mentions in "Skills" section
  - Long-form story-based resumes may lose narrative context
- **No Temporal Weighting**: Recent experience vs. older experience scored equally
- **Quantification Bias**: Roles with metrics (revenue, growth) may be overweighted

### 4. **Keyword Analysis Limitations**
- **Exact Matching Only**: POS-based keyword extraction is sensitive to word forms
  - "Kubernetes" and "K8s" are treated as different keywords
  - Plural vs. singular forms are distinguished (might miss "databases" if looking for "database")
- **Stopword Exclusion**: May filter out important terms that happen to be less common adjectives
- **Limited Synonym Recognition**: No built-in synonym matching for related concepts

### 5. **Scoring Edge Cases**
- **Scale Sensitivity**: Cosine similarity scores aren't calibrated to industry standards
  - A score of 0.65 doesn't have an inherent interpretation (excellent vs. moderate vs. poor)
  - Scores vary significantly by job type and document length
- **No Weighting System**: All resume sections treated equally
  - Skills may be underweighted vs. Experience for technical roles
  - Contact info is removed but other sections aren't prioritized
- **Short Document Impact**: Minimal job descriptions or resumes may yield unreliable scores

### 6. **AI Recommendation Limitations**
- **Requires API Key**: Gemini API calls cost money; feature depends on external service availability
- **Hallucination Risk**: Despite guardrails, LLMs may occasionally suggest unrealistic changes
- **No Verification**: System doesn't validate that generated recommendations are actually implementable
- **Generic Structure**: Recommendations follow a fixed format; may not capture unique role nuances

### 7. **Privacy & Security**
- **Data Processing**: Resume and job description text are processed locally, but:
  - Text is sent to Google Gemini API if recommendations are requested
  - No persistent logging of documents, but check Google's data policies
- **API Key Exposure**: Users must provide their Gemini API key; ensure it's kept secure in environment variables

### 8. **Performance & Scalability**
- **Single Document Processing**: Designed for 1-1 resume-to-job matching
  - Batch processing (e.g., matching 1 resume to 100 jobs) requires optimization
- **Model Size**: SentenceTransformer model (~80 MB) must load into memory
- **Latency**: Embedding generation for long documents may take 5-15 seconds

## 🔮 Future Improvements

- [ ] Support for multiple resume formats (DOCX, HTML, plain text)
- [ ] Batch processing interface for multiple job opportunities
- [ ] Fine-tuned embedding models for specific industries (finance, healthcare, tech)
- [ ] Historical tracking of resume versions and improvements over time
- [ ] Integration with LinkedIn and job boards for automated job matching
- [ ] Advanced visualization of semantic space (t-SNE projections)
- [ ] Industry-specific keyword weighting and benchmarking
- [ ] Support for video resume analysis (experimental)
- [ ] Multi-language support
- [ ] Resume template recommendations based on industry

## 📝 Usage Tips

1. **For Best Results:**
   - Use well-structured, standard resume formats
   - Ensure job descriptions are complete and untruncated
   - Use recent embeddings model versions for latest NLP capabilities
   - Keep resumes to 1-2 pages for optimal processing

2. **Interpreting Scores:**
   - Overall score > 0.75: Strong semantic alignment, resume is well-tailored
   - Overall score 0.5-0.75: Moderate alignment, targeted improvements recommended
   - Overall score < 0.5: Significant gaps, consider whether the role is a good fit

3. **Using Recommendations:**
   - The AI suggestions are starting points, not final copy
   - Verify all facts and metrics before updating your resume
   - Tailor recommendations further to match your authentic experience
   - Ensure changes don't introduce factual inaccuracies

## 🤝 Contributing

Feedback and contributions are welcome. Areas for improvement:
- Enhanced resume parsing logic
- Additional scoring metrics
- Integration with ATS systems
- Multi-language support
- Performance optimizations

## 📄 License

This project is provided as-is for educational and personal use.

## ❓ FAQ

**Q: Is my resume data stored or shared?**
A: Resumes are processed in-memory only. If you use the AI recommendations feature, text is sent to Google's Gemini API. Check Google's data retention policies for details.

**Q: Can I use this for multiple job applications?**
A: Yes, you can upload the same resume and test it against different job descriptions repeatedly.

**Q: What if my resume format is unusual?**
A: The system works best with standard resume formats. If parsing fails, try reformatting sections with clear headers.

**Q: How accurate are the scores?**
A: Scores are relative indicators of semantic similarity, not absolute measures. Use them to identify patterns and areas for improvement, not as binary accept/reject decisions.

**Q: Can I improve my score?**
A: Yes! Use the AI recommendations to incorporate missing keywords naturally into your resume, and ensure your actual experience aligns with what you're claiming.

**Q: Does this guarantee I'll get an interview?**
A: No. A high match score improves your odds, but many factors beyond resume alignment affect hiring decisions. Use this tool as one part of your job search strategy.

**Q: What about resume screening by ATS systems?**
A: This tool optimizes for semantic alignment and keyword coverage, which helps with both ATS systems and human reviewers. However, always follow application instructions and ATS guidelines in job postings.

---

**Last Updated:** February 2026
