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
model="llama-3.3-70b-versatile"
message = "Hello, Groq! Can you tell me about recipe to make fried rice?"
role = "user"

response = client.chat.completions.create(
    model=model,
    messages=[{"role":role, "content": message}]
)

print(f"Response: {response}")
print("**********Message Content**********")
print(f"Response: {response.choices[0].message.content}")