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

def llm_ans (prompt):
    message= {
        "role":"user",
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    ans = response.choices[0].message.content
    return ans


bad_prompt = """This is a user complaint. My laptop is not working. Classify this"""

# bad_prompt_answer = llm_ans(bad_prompt)
# print(bad_prompt_answer)

role_prompt = """
#ROLE: You are a support assistant at a mobile/laptop company
This is a user complaint. My laptop is not working. Classify this
"""

# role_prompt_answer = llm_ans(role_prompt)
# print(role_prompt_answer)

task_prompt = """
#ROLE: You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
This is a user complaint. My laptop is not working.
"""

# task_prompt_answer = llm_ans(task_prompt)
# print(task_prompt_answer)

constraint_prompt = """
#ROLE: You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of the three issues - technical, billing, return
This is a user complaint. My laptop is not working.
"""

# constraint_prompt_answer = llm_ans(constraint_prompt)
# print(constraint_prompt_answer)

output_prompt = """
#ROLE: You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of the three issues - technical, billing, return
#OUTPUT FORMAT
Your answer should be only in one word only. The answer should be one of the categories given in the constraints
This is a user complaint. My laptop is not working.
"""

# output_prompt_answer = llm_ans(output_prompt)
# print(output_prompt_answer)


example_prompt = """
#ROLE: You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of the three issues - technical, billing, return
#OUTPUT FORMAT
Your answer should be only in one word only. The answer should be one of the categories given in the constraints
#EXAMPLE
For example, if the user writes the complain - my laptop is not working, i want a return. then the classification is return.
This is a user complaint. My laptop is not working.
"""

# example_prompt_answer = llm_ans(example_prompt)
# print(example_prompt_answer)


fallback_prompt = """
#ROLE: You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of the three issues - technical, billing, return
#OUTPUT FORMAT
Your answer should be only in one word only. The answer should be one of the categories given in the constraints
#EXAMPLE
For example, if the user writes the complain - my laptop is not working, i want a return. then the classification is return.
#FALLBACK
If the issue is unrelated to any of the above mentioned contraints value, return other. if the issue is not related to any laptop or mobile then return "Sorry, Please mention the problems related to mobile and phone"
This is a user complaint. My laptop is not working.
"""

fallback_prompt_answer = llm_ans(fallback_prompt)
print(fallback_prompt_answer)