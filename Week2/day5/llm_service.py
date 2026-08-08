import os
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# print(f"API Key: {GROQ_API_KEY}")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=GROQ_API_KEY)
model = "llama-3.3-70b-versatile"


def llm_call(messages):
        

    # if user_prompt != "":
    #     messages.append({
    #         "role": "user",
    #         "content": user_prompt
    #     })

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content

    return answer
