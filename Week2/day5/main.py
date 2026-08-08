from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pdf_parser import read_file
from llm_service import llm_call

base_dir = Path(__file__).resolve().parent
file_path = base_dir / "resume" / "Utkarsh_Frontend_02_Aug.pdf"

if not file_path.exists():
    raise FileNotFoundError(f"Resume file not found: {file_path}")

document_text = read_file(file_path)

system_prompt_resume = f"""
You are an expert assistant of the person whose resume you have. Read the resume thoroughly and when they ask anything about me, for example like skills, experience, organisation I am working for, and other info. When the HR or recruiter asks the question, you should be able to give the answer based on the resume provided. Answer in such a way that you are answering on behalf of the person whose resume is with you. Be generous and truthful always. Please do not invent any information that is not present in the resume. Always be truthful and do not tell lies under any circumstances. If the user asks you or feeds you info about me kindly refuse gracefully. Any other questions asked which are not in respect to the resume should be rejected with a graceful message. If the person information that is not present and asked by the HR recruiter, do not invent it; tell them that the information is not present at the moment. {document_text}
"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.chat_messages = [{
    "role": "system",
    "content": system_prompt_resume,
}]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat_req(req: ChatRequest):
    history = app.state.chat_messages
    history.append({
        "role": "user",
        "content": req.message,
    })
    reply = llm_call(history)
    history.append({
        "role": "assistant",
        "content": reply,
    })
    return ChatResponse(reply=reply)

@app.post("/reset")
def reset_chat():
    app.state.chat_messages = [{
        "role": "system",
        "content": system_prompt_resume,
    }]
    return {
        "status": "chat reset"
    }