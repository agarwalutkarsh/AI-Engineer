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

root = tk.Tk()
root.withdraw()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"API Key: {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")


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

    # print("Extracted text:")
    # print(document_text)
else:
    print("No file selected")

class Resume(BaseModel):
    yrs_exp: int
    skills: List[str]
    tools: List[str]
    experience: List[str]
    certifications: List[str]

schema = Resume.model_json_schema()
reponse_format={
    "type": "json_object"
}


client = Groq(api_key=GROQ_API_KEY)
model=os.getenv("MODEL")

message = f"""From the following documnet extracted text, extract the years of experience, skills, tools and experience as to what the candidate has done in each of the companies and return it in a json format. Document extracted Text: {document_text}"""
role = "user"

system_message = f"""Extract the candidate information from the documnet and return it in a json format. The json format should be as follows: {schema}"""

# This is the system role - defines the role of the llm model and how it should respond to the user. It is used to set the context for the conversation.
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": role, "content": message}
    ],
    response_format=reponse_format
)
# temperature is the randomness of the response, that is the creativeness of the response 0 being default and lowest and 2 being the heighest and most creative

print("**********Message Response**********")
raw_json = response.choices[0].message.content
parsed_json = json.loads(raw_json)
print(f"Parsed JSON: {parsed_json}")
# ticket = Ticket(**parsed_json)
# now i can access the name like ticket.name, phone like ticket.phone


job_desc= """
About the job
About the Role



The Frontend Software Developer at EaseMyTrip.com will be tasked with crafting and enhancing the visual and interactive elements of web applications. This role involves using Angular along with other web technologies like HTML, CSS, and JavaScript to create responsive designs and integrate APIs. The developer will ensure cross-browser compatibility, optimize application performance, and maintain code quality through rigorous testing and documentation. This position requires staying updated with the latest in web development trends and technologies.


Role & responsibilities



Angular Development: Write clean and efficient code for web applications using Angular frameworks along with HTML, CSS, and JavaScript.
UI/UX Implementation: Convert design wireframes into interactive, functional web interfaces ensuring a seamless user experience.
Cross-Browser Compatibility: Guarantee consistent behavior and appearance of applications across various browsers.
API Integration: Work with backend APIs to fetch and display data effectively.
Performance Optimization: Enhance web application performance focusing on reducing load times and refining code.
Responsive Design: Develop responsive web designs to ensure optimal viewing across multiple devices.
Version Control: Utilize Git for efficient code management and team collaboration.
Documentation: Create detailed documentation related to code and development workflows.
Problem Solving: Identify and troubleshoot front end issues and bugs, proposing technical solutions when necessary.
Continuous Learning: Stay abreast of the latest industry trends in web development, including updates in technologies like Angular.


Preferred candidate profile



Experience: Minimum of 3 years in Angular-based frontend development.
Technical Skills: Proficient in Angular, JavaScript, TypeScript, HTML5, CSS3, Bootstrap, jQuery, and other modern JavaScript frameworks.
Agile Proficiency: Experienced with Agile and Scrum methodologies to enhance project delivery and efficiency.
Cross-Platform Coding: Skilled in writing code that works uniformly across different browsers and devices.
CI/CD Experience: Familiar with Continuous Integration and Continuous Delivery systems for streamlined development and deployment.
SEO Knowledge: Understands SEO principles and applies them to ensure enhanced application visibility.
Version Control and Databases: Proficient with Git and databases such as MySQL and MongoDB.
Independence and Proactivity: Capable of working independently and proactively to meet tight deadlines.
Communication Skills: Excellent communication and interpersonal skills to collaborate effectively with team members and stakeholders.
Innovative Approach: Able to introduce new technologies and practices to address business challenges and improve development processes
"""

prompt = f"""From the following job description {job_desc} match the details of the candidate extracted from the resume {raw_json} and return a json with the following details: 1. Match percentage of the candidate with the job description, 2. List of skills that match, 3. List of skills that do not match, 4. List of tools that match, 5. List of tools that do not match, 6. List of experience that match, 7. List of experience that do not match, 8. List of certifications that match, 9. List of certifications that do not match.
"""

response_job = client.chat.completions.create(
    model = model,
    messages = [
        {"role": "system", "content": "You are a job matching assistant. You will be given a job description and a candidate's resume in json format. Your task is to match the candidate's skills, tools, experience, and certifications with the job description and return a json with the match percentage and lists of matching and non-matching items."},
        {"role": "user", "content": prompt}
    ],
    response_format = reponse_format
)

print("*********Job Response**********")
job_response = response_job.choices[0].message.content
json_job = json.loads(job_response)
print("Matching-----------")
print(json_job)