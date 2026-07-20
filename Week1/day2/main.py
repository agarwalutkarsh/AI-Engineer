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
# message = "Hello, Groq! Can you tell me about recipe to make fried rice?"
message = "Hello, Groq! Can you tell me about scrum call updates as i wasnt available for the call?"
role = "user"

system_message = "You are my manager at office and you are very strict and serious. Only entertain the questions related to work and office."

# This is the system role - defines the role of the llm model and how it should respond to the user. It is used to set the context for the conversation.
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": role, "content": message}
    ],
    temperature = 2
)
# temperature is the randomness of the response, that is the creativeness of the response 0 being default and lowest and 2 being the heighest and most creative

# print(f"Response: {response}")
print("**********Message Content**********")
print(f"Response: {response.choices[0].message.content}")