from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#1. Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

#2. Sample documents to index
documents = [
    "The cat sits on the mat.",
    "Dogs are great pets.",
    "I love programming in Python.",
    "Artificial Intelligence is the future.",
    "OpenAI develops advanced AI models."
]

#3. Generate embeddings for the documents
doc_embeddings = model.encode(documents)

dimesion = doc_embeddings.shape[1]

#4. Create a FAISS index
index = faiss.IndexFlatL2(dimesion) #L2 We'll treat smaller distance as more similar
index.add(doc_embeddings)
print(f"Number of documents indexed: {index.ntotal}")

#5. Sample query
def search(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True )
    distances, indices = index.search(query_embedding, k)

    print(f"Query: {query}")
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        print(f"Rank: {rank} with distance: {distance}: Document: '{documents[idx]}'")

#6. Perform search with a sample query
search("I enjoy coding in Python.")
search("Pets are wonderful companions.")
search("Advancements in AI technology.")