import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = Path("data")

# 2. Load all .txt / .md files
documents = []
metadatas = []  # (filename, chunk_id)

def load_files():
    for path in DATA_DIR.glob("**/*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # simple chunking: split into paragraphs
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"file": str(path), "chunk": i})

load_files()
print(f"Loaded {len(documents)} chunks from {len(set(m['file'] for m in metadatas))} files.")

# 3. Embed chunks
embeddings = model.encode(documents, convert_to_numpy=True)
dimension = embeddings.shape[1]

# 4. Build FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Indexed {index.ntotal} chunks.")

def search(query, k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)

    print(f"\nQuery: {query}")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        meta = metadatas[idx]
        print(f"  {rank}. (dist={dist:.4f}) file={meta['file']} chunk={meta['chunk']}")
        print(f"     {documents[idx][:200]}")  # show first 200 chars

# 5. Simple interactive loop
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        search(q, k=3)
