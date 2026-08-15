import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"API Key: {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=GROQ_API_KEY)
model=os.getenv("MODEL")

knowledge_base = {
    "place": "I am going to visit Indonesia",
    "days": "For 10 days i will be visiting."
}

def retrieve (question) :
    if ("place" in question):
        return knowledge_base["place"]
    elif ("days" in question) :
        return knowledge_base["days"]
    else :
        return None

def ask_llm (question) :
    context = retrieve(question)
    system_prompt = f"""Answer in one line only. Answer within the context only. {context}"""
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    message = {
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(
        model=model,
        messages = messages
    )
    return response.choices[0].message.content

print(ask_llm("What is Mount Bromo famous for?"))
print(ask_llm("For how many days i am going to visit?"))
print(ask_llm("which place i am going to visit?"))

# At core RAG is passing the context to the LLM by retreiving similar data from knowledge base and getting a resoonse out of it.
# this is the 1st iteration of RAG. where we are giving the context to the llm by retreiving the similar information by a retreiving function and llm responds from that.
# This has many problems - if i write how long i am going there it will fail as it doesn't have that context but the question is similar.
# If i will be mentioning all the synonyms it will be lot of if else
