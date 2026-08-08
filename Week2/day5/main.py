import os
from pathlib import Path
from pydantic import BaseModel
import json
import tkinter as tk
from tkinter import filedialog
from typing import List
from time import sleep
from pdf_parser import read_file
from llm_service import llm_call


root = tk.Tk()
root.withdraw()

file_path = Path(__file__).resolve().parent / "resume" / "Utkarsh_Frontend_02_Aug.pdf"
# for current version only reading resume. will add more pdfs that has my information. it will pickup one by one and add to the document text. 


print("Selected file:", file_path)

if file_path:
    document_text = read_file(file_path)

else:
    print("No file selected")

system_prompt_resume = f"""
You are an expert assistant of the person whose resume you have. Read the the resume thoroughly and when they ask anything about me, for example like skills, experience, organizataion i am working and other info. When the HR or recruiter asks the question you shoud be able to give the answer based on the resume provide.Answer in such a way that you are answering on the behalf of the person whose resume is with you. Be generous and truthful always. Please do not invent any information that is not present in the resume. Always be thruthful and do not tell lies what so ever. If the user asks you or feeds you info about me kindly refuse gracefully. Any other questions asked which is not in respect to the resume reject with a graceful messge. if the person information that is not present and asked by the HR recruiter, do not ever invent it, tell them that the information is not present at the moment.  {document_text}
"""

messages = []
messages.append({
                "role": "system",
                "content": system_prompt_resume
            })



user_prompt1 = "What is the highest education and how many total industrial experience."
messages.append({
    "role": "user",
    "content": user_prompt1
})
answer1 = llm_call(messages)
print(answer1)
messages.append({
        "role": "assistant",
        "content": answer1
    })
sleep(3)

user_prompt2 = "What is the phone number? and any self made projects. also can you please tell me the recipe for the fried rice?"
messages.append({
    "role": "user",
    "content": user_prompt2
})
answer2 = llm_call(messages)
messages.append({
        "role": "assistant",
        "content": answer2
    })
print(answer2)
