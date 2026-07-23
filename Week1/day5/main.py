import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
import json
from pypdf import PdfReader
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"API Key: {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")


class Ticket(BaseModel):
    name: str
    phone: str
    email: str
    issue: str

schema = Ticket.model_json_schema()
reponse_format={
    "type": "json_object"
}

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


file_path = filedialog.askopenfilename(
    title="Select a PDF or Word file",
    filetypes=[
        ("PDF files", "*.pdf"),
        ("Word files", "*.docx"),
        ("All files", "*.*")
    ]
)

print("Selected file:", file_path)

if file_path:
    document_text = read_file(file_path)

    print("Extracted text:")
    print(document_text)
else:
    print("No file selected")

complaint = "I am Utkarsh, I have a complaint regarding the product I purchased. It is not functioning as expected and I would like to request a refund or a replacement. Please let me know the process for returning the product and getting my issue resolved. My email is abc@test.com and my phone number is 1234567890. Thank you."

client = Groq(api_key=GROQ_API_KEY)
model="llama-3.3-70b-versatile"

message = f"""From the following complaint, extract the name, phone number and email address and issue of the customer and return it in a json format. Complaint: {complaint}"""
role = "user"

system_message = f"""Extract the personal information from the complaint and return it in a json format. The json format should be as follows: {schema}"""

# This is the system role - defines the role of the llm model and how it should respond to the user. It is used to set the context for the conversation.
# response = client.chat.completions.create(
#     model=model,
#     messages=[
#         {"role": "system", "content": system_message},
#         {"role": role, "content": message}
#     ],
#     response_format=reponse_format
# )
# # temperature is the randomness of the response, that is the creativeness of the response 0 being default and lowest and 2 being the heighest and most creative

# print("**********Message Response**********")
# raw_json = response.choices[0].message.content
# parsed_json = json.loads(raw_json)
# print(f"Parsed JSON: {parsed_json}")
# ticket = Ticket(**parsed_json)
# now i can access the name like ticket.name, phone like ticket.phone
