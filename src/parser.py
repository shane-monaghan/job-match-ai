import pdfplumber
import unicodedata
import re

from typing import Dict

def clean_text(text: str) -> str:
    """
    Normalizes and cleans extracted text by removing non-ASCII characters.

    This function applies NFKD normalization to decompose combined characters 
    (like accents) and then strips away any characters that cannot be 
    represented in standard ASCII (such as emojis or complex PDF artifacts).

    Args:
        text (str): The raw text string extracted from the PDF.

    Returns:
        str: A cleaned, ASCII-only version of the input text.
    """
    # Decompose combined characters (e.g., 'é' becomes 'e' + '´') 
    # so that the base letter can be preserved during ASCII encoding.
    normalized_text = unicodedata.normalize("NFKD", text)

    # Convert to ASCII bytes, 'ignore' removes characters that don't fit the schema.
    # This effectively deletes artifacts like the '' symbol.
    clean_bytes = normalized_text.encode("ascii", "ignore")

    # Convert the filtered bytes back into a standard Python string.
    return clean_bytes.decode("utf-8")

def extract_text(resume_path : str) -> str:
    """
    Extracts all text content from a PDF resume file.

    Iterates through each page of the PDF, extracts text using pdfplumber, 
    and concatenates it into a single normalized string.

    Args:
        resume_path (str): The local file path or file-like object of the resume PDF.

    Returns:
        str: A single string containing the combined text of all pages.
    """
    entire_text = ""

    with pdfplumber.open(resume_path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            # extract_text() can return None if a page is an image or empty
            page_text = page.extract_text()

            if page_text:
                entire_text += page_text + '\n'

    # .strip() removes leading/trailing whitespace from the final document 
    entire_text = entire_text.strip()

    return clean_text(entire_text)

def chunk_section(header : str, content : str, max_tokens : int, overlap : int) -> list[str]:
    content_tokens = content.split()

    if len(content_tokens) <= max_tokens:
        return [f"{header}: {content}"]
    
    step_size = max_tokens - overlap
    chunks = []

    for i in range(0, len(content_tokens), step_size):
        segment = content_tokens[i : i + max_tokens]

        prefix = f"{header}:" if i == 0 else f"{header} (Continued):"
        chunk_text = f"{prefix} {' '.join(segment)}"
        
        chunks.append(chunk_text)

        if i + max_tokens >= len(content_tokens):
            break
    
    return chunks

def chunk_resume(text : str) -> Dict[str, str]:
    header_pattern = r"^\s*(SUMMARY|OBJECTIVE|EXPERIENCE|EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|AWARDS|ADDITIONAL EXPERIENCE|SKILLS AND CERTIFICATIONS)\s*$"

    sections = re.split(header_pattern, text, flags=re.MULTILINE | re.IGNORECASE)

    resume_sections = {}
    if sections:
        resume_sections["CONTACT INFO"] = [sections[0].strip()]

    for i in range(1, len(sections), 2):
        header = sections[i].strip().upper()
        content = sections[i+1].strip()

        chunks = chunk_section(header, content, max_tokens=150, overlap=20)

        resume_sections[header] = chunks
    
    return resume_sections


if __name__ == "__main__":
    test_path = "data/Resume.pdf" 
    
    # 1. Extract and Clean
    print("--- Extracting Text ---")
    raw_text = extract_text(test_path)
    
    # 2. Chunk by Section and Tokens
    print("--- Chunking Resume ---")
    structured_resume = chunk_resume(raw_text)
    
    # 3. Display Results
    print(f"\nFound {len(structured_resume)} distinct sections.\n")
    
    for section, chunks in structured_resume.items():
        print(f"[{section}] - {len(chunks)} chunk(s)")
        for idx, c in enumerate(chunks):
            # Print a snippet of each chunk to verify the Header injection
            snippet = c[:100].replace('\n', ' ')
            print(f"  Chunk {idx+1}: {snippet}...")
        print("-" * 30)