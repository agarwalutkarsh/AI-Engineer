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
message = "Hello, Groq! Can you tell me about moon? Answer in 500 words"
role = "user"

stream = client.chat.completions.create(
    model=model,
    messages=[{"role":role, "content": message}],
    stream=True # by default it is false - true when the answer is read to user.
)

print("**********Message Content**********")
# Stream - instead of printing the entire result in one go, if we want to print the line as soon as it gets generated we use stream. it is used to increase user experience and reduce the response time
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content, end="", flush=True)
