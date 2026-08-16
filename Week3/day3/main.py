import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2') # 384 vector size - array size of features
groq_model = os.getenv("MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

if not GROQ_API_KEY:
    raise ValueError("API key not found")

client = Groq(api_key = GROQ_API_KEY)


# Knowledge Base - array of string. each string means one document
# embedding will create the vectors for all the the strings in this knowledge base.
documents = [
    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday. Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]

documents_embedding = model.encode(documents)

# def cosine_similarity (a,b):
#     return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
def cosine_similarity(vec1, vec2):
    # np.linalg.norm(vec1)   Ensure the vectors are not zero vectors
    # np.linalg.norm(vec2)   Ensure the vectors are not zero vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# calculate the similarity between the query embedding and each documnet embedding.
# take the score and the heighest score vector should point to the most relevant context

def retrieval(query_embedding):
    #score array for maintaing the scores of the knowledge base
    score = []
    for i,document_vector in enumerate(documents_embedding): # enumerate converts the simple array tp the array where each element has the index also
        # [a, b, c] => [(0,a), (1,b), (2,c)]
        similarity = cosine_similarity(query_embedding, document_vector)
        score.append((similarity, documents[i])) # stores the score and the line for which the score is calculated
    score.sort(reverse=True) # sort in reverse so that the 0th index is the most similar one
    print(score[0])
    return (score[0][1])


def ask_llm(query):
    query_embedding = model.encode(query)
    context = retrieval(query_embedding)
    sys_prompt = f"""The following is the context {context}. Answer on the basis of context. And answer in one line"""
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]

    response = client.chat.completions.create(
        model=groq_model,
        messages=messages
    )

    return response.choices[0].message.content

# create the embedding for the query
# query = "How much vacation can i get?"
query = " is there any reimbursement?"

print(ask_llm(query))