import os
import math
from openai import OpenAI

# 1. Set up client (Reads API key from environment variable OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def consine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

text = [
    "I love studying vector embeddings and RAG systems.",
    "Understanding retrieval augmented generation is really fun.",
    "Yesterday I cooked pasta and watched a movie.",
]

# 2. Create embeddings for the text data
response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

embeddings = [data.embedding for data in response.data]

# 3. Compute cosine similarity between the first and other embeddings
sim_0_1 = consine_similarity(embeddings[0], embeddings[1])
sim_0_2 = consine_similarity(embeddings[0], embeddings[2])
sim_1_2 = consine_similarity(embeddings[1], embeddings[2])

print("Text 0:", text[0])
print("Text 1:", text[1])
print("Text 2:", text[2])
print()
print(f"Cosine similarity between text[0] and text[1]: {sim_0_1}")
print(f"Cosine similarity between text[0] and text[2]: {sim_0_2}")
print(f"Cosine similarity between text[1] and text[2]: {sim_1_2}")