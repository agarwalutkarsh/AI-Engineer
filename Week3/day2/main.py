import os
from pathlib import Path
# from dotenv import load_dotenv
# from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    # np.linalg.norm(vec1)   Ensure the vectors are not zero vectors
    # np.linalg.norm(vec2)   Ensure the vectors are not zero vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

model = SentenceTransformer('all-MiniLM-L6-v2') # 384 vector size - array size of features
# text = "This is a test sentence for embedding."

# embedding = model.encode(text)
# print(f"""
# Embedding Shape: {embedding.shape}
# """)
# print(f"""
# Embedding first 10 values: {embedding[:10]}
# """)

v1 = model.encode("There are 24 paid leaves")
v2 = model.encode("i can take 24 paid days off in a year")
print(f"Cosine Similarity: {cosine_similarity(v1, v2)}") # similarity score between two sentences 1 highest similarity and 0 lowest similarity