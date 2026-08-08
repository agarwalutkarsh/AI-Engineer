import os
from pathlib import Path
from pypdf import PdfReader

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def read_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return read_pdf(file_path)

    # elif extension == ".docx":
    #     return read_docx(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Only PDF and DOCX files are supported."
        )

# file_path = filedialog.askopenfilename(
#     title="Select a PDF or Word file",
#     filetypes=[
#         ("PDF files", "*.pdf")
#     ]
# )