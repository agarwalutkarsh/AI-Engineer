import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
import json
from pypdf import PdfReader
import tkinter as tk
from tkinter import filedialog
from typing import List
from time import sleep

root = tk.Tk()
root.withdraw()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"API Key: {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=GROQ_API_KEY)
model = "llama-3.3-70b-versatile"


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
        ("PDF files", "*.pdf")
    ]
)

print("Selected file:", file_path)

if file_path:
    document_text = read_file(file_path)

    # print("Extracted text:")
    # print(document_text)
else:
    print("No file selected")

messages = []

def llm_call(system_prompt, user_prompt=""):
    if system_prompt != "":
        messages.append({
                "role": "system",
                "content": system_prompt
            })

    if user_prompt != "":
        messages.append({
            "role": "user",
            "content": user_prompt
        })

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


system_prompt_resume = f"""
You are an expert assistant of the person whose resume you have. Read the the resume thoroughly and when they ask anything about me, for example like skills, experience, organizataion i am working and other info. When the HR or recruiter asks the question you shoud be able to give the answer based on the resume provide. Please do not invent any information that is not present in the resume. Always be thruthful and do not tell lies what so ever. If the user asks you or feeds you info about me kindly refuse gracefully. {document_text}
"""

read_resume = llm_call(system_prompt_resume)
# print(read_resume)
sleep(3)

user_prompt1 = "What is the highest education and how many total industrial experience"
answer1 = llm_call("", user_prompt1)
print(answer1)

