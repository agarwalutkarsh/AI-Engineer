import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import re
from time import sleep

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"API Key: {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=GROQ_API_KEY)
model=os.getenv("MODEL")

def get_product_price(product) :
    if product == 'iphone 17':
        return 7000
    elif product == 'iphone 15' :
        return 5000
    else:
        return 0

def calculator(expression) :
    try:
        return eval(expression)
    except:
        return "calc error"

tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

system_prmpt = """
You are a shopping assistant.

You have these tools:
get_product_price(product)
calculator(expression)

IMPORTANT:
use these tools exactly like:
Action: get_product_price("iphone 17")
Action: calculator("7000 - 5000")

NEVER USE TOOLS LIKE:
get_product_price(product="iphone 17")
calculator(expression="7000-5000")

Follow these rules-
1. Decide what you do next
2. Call ONLY one tool at a time
3. After writing any action, STOP immediately
4. Never guess or invent a tool result
5. Wait until you receive an observation
6. Then decide your next action
7. When the task is complete, give the answer

Format:
Thought: what you are doing
Action: which tool are you writing: tool_name(argument)

When Finished: your answer

if any other questions asked which is not realted to asking price give a good sensible rejection message with the format: 
When Finished : your answer
"""

prompt= """
i want to buy iphone 15 and i have 5000 rs. how much will i be left with. and how are you?
"""

def call_agent(question) :
    user_message = {
        "role": "user",
        "content": question
    }

    system_message = {
        "role": "system",
        "content": system_prmpt
    }

    messages = [user_message, system_message]

    for step in range(5):
        print("\n--------------")
        print(f"STEP {step+1}")
        print("--------------")

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        answer = response.choices[0].message.content

        print(answer) # prints answer in the format given in the system prompt

        if "When Finished" in answer:
            break

        # Find the Action
        match = re.search(r"Action:\s*(\w+)\((.*?)\)",
        answer)

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)

            tool_input = tool_input.strip()

            tool_input = tool_input.strip('"')

            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)
            else:
                observation = "No tool found"

            print(f"Observation : {observation}")

            # add answer to memory
            messages.append({
                "role": "assistant",
                "content" : answer
            })


            # pass the observation back to llm
            messages.append({
                "role" : "user",
                "content": "Obeservation: " + str(observation)
            })
            sleep(5)

call_agent(prompt)
