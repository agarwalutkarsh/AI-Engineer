import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, PayloadSchemaType, MatchValue


load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2') # 384 vector size - array size of features
groq_model = os.getenv("MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_URL=os.getenv("QDRANT_URL")
print(GROQ_API_KEY)

if not GROQ_API_KEY:
    raise ValueError("API key not found")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
print("Connected to Qdrant Client")

groq_client = Groq(api_key = GROQ_API_KEY)


COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384


if client.collection_exists(COLLECTION_NAME):
    print("Deleting the previously existing collection")
    client.delete_collection(COLLECTION_NAME)



client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(
        size = EMBEDDING_SIZE,
        distance = Distance.COSINE
    )
)

print(f"""Created collection {COLLECTION_NAME} with size {EMBEDDING_SIZE} and Cosine""")

client.create_payload_index(
    collection_name = COLLECTION_NAME,
    field_name = "category",
    field_schema = PayloadSchemaType.KEYWORD
)

reimbursement_filter = Filter(
    should = [
        FieldCondition(
            key="category",
            match = MatchValue(value = "reimbursement")
        ),
        FieldCondition(
            key="category",
            match = MatchValue(value = "leave")
        )
    ]
)

with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [document["text"] for document in documents]

embeddings = model.encode(texts)

print(f"""Generated {len(embeddings)} embeddings""")

points = []

for i in range(len(documents)):
    point = PointStruct(
        id = i + 1,
        payload = documents[i],
        vector = embeddings[i].tolist()
    )
    points.append(point)

client.upsert(
    collection_name = COLLECTION_NAME,
    points = points
)

print(f"""Uploaded {len(points)} to Qdrant""")


def search_filter(query, top_k = 3, query_filter = None):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        query_filter = query_filter,
        with_payload = True
    ).points

    return results

query = "How much leaves does an employee get and is there any reimbursement?"

results = search_filter(query, 3, reimbursement_filter)

print("\nSerach Results:")
for result in results:
    print(f"""{result.score:.3f}""")
    print(f"""{result.payload["text"]}""")
    print()


def ask_llm(query, context):
    prompt = f"""This is the context {context} and the question {query} respectively. answer on the basis of the context only do not invent any new information on your own."""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = groq_client.chat.completions.create(
        model = groq_model,
        messages = messages
    )

    return response.choices[0].message.content

context = "\n".join(
    result.payload["text"]
    for result in results
    )

print("\nFINAL ANSWER -----")
print(ask_llm(query, context))