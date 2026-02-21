import pdfplumber

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

    # Replace unknown unicode
    entire_text = entire_text.replace('\uf0b7', '-') 
    entire_text = entire_text.replace('\u2022', '-') # Standard unicode bullet

    # .strip() removes leading/trailing whitespace from the final document            
    return entire_text.strip()
            
if __name__ == "__main__":
    # This only runs if you run this file directly
    test_path = "data/Kayla_Yi_Resume.pdf" 
    print(extract_text(test_path))