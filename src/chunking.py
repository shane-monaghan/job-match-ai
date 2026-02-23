import re

def chunk_section(header: str, content: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Splits a long string into smaller chunks with a specified overlap.

    Each chunk is prefixed with the header to maintain context. Subsequent 
    chunks after the first include a "(Continued)" suffix in the header.

    Args:
        header: The title or identifier of the section.
        content: The raw text body to be split into chunks.
        max_tokens: The maximum number of words allowed per chunk.
        overlap: The number of words to repeat from the previous chunk 
            to maintain semantic continuity.

    Returns:
        A list of strings, where each string is a formatted chunk of text.
    """
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

def chunk_resume(text: str) -> dict[str, list[str]]:
    """
    Parses a resume string into sections and further chunks the content.

    Uses a predefined list of common resume headers to split the text. 
    The initial segment before any header is categorized as 'CONTACT INFO'. 
    Each identified section is then subdivided into smaller overlapping 
    chunks using `chunk_section`.

    Args:
        text: The full text content of the resume to be processed.

    Returns:
        A dictionary where keys are section headers (e.g., 'EXPERIENCE') 
        and values are lists of string chunks belonging to that section.

    Note:
        This function relies on a global or previously defined `chunk_section` 
        function to handle the token-based splitting of section bodies.
    """
    header_pattern = r"^\s*(SUMMARY|OBJECTIVE|EXPERIENCE|EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|AWARDS|ADDITIONAL EXPERIENCE|SKILLS AND CERTIFICATIONS)\s*$"

    # Splits the text while capturing the headers due to the parentheses in the pattern
    sections = re.split(header_pattern, text, flags=re.MULTILINE | re.IGNORECASE)

    resume_sections = {}
    if sections:
        # Initial text before the first regex match is treated as contact info
        resume_sections["CONTACT INFO"] = [sections[0].strip()]

    # Iterate through matches: re.split with capturing groups returns [prefix, match, suffix, match, suffix...]
    for i in range(1, len(sections), 2):
        header = sections[i].strip().upper()
        content = sections[i+1].strip()

        # Breaks down the specific section into manageable chunks for LLM processing
        chunks = chunk_section(header, content, max_tokens=150, overlap=20)

        resume_sections[header] = chunks
    
    return resume_sections