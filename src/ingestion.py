import pdfplumber
import unicodedata

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

