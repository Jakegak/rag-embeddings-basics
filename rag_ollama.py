import os
from pathlib import Path
import json
import requests

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------ CONFIG ------------
DATA_DIR = Path("data")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
# -------------------------------

model = SentenceTransformer(EMBED_MODEL_NAME)

documents: list[str] = []
metadatas: list[dict] = []

def load_files():
    for path in DATA_DIR.glob("**/*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"file": str(path), "chunk": i})

def build_index():
    load_files()
    if not documents:
        raise RuntimeError("No documents loaded from data/")

    print(f"Loaded {len(documents)} chunks from {len(set(m['file'] for m in metadatas))} files.")
    embeddings = model.encode(documents, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} chunks.")
    return index

index = build_index()

def retrieve(query: str, k: int = 4):
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = metadatas[idx]
        results.append({
            "distance": float(dist),
            "text": documents[idx],
            "file": meta["file"],
            "chunk": meta["chunk"],
        })
    return results

def build_prompt(query: str, contexts: list[dict]) -> str:
    context_block = ""
    for i, ctx in enumerate(contexts, start=1):
        context_block += (
            f"[{i}] (file={ctx['file']} chunk={ctx['chunk']})\n"
            f"{ctx['text']}\n\n"
        )

    prompt = (
        "You are a helpful assistant answering questions based ONLY on the given context.\n"
        "If the answer is not in the context, say you don't know and do NOT hallucinate.\n\n"
        f"Context:\n{context_block}"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt

def answer_with_ollama(query: str, contexts: list[dict]) -> str:
    prompt = build_prompt(query, contexts)
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a grounded RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    # Ollama chat API typically: { "message": { "content": "..." }, ... }
    return data["message"]["content"].strip()

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        hits = retrieve(q, k=4)
        print("\n--- RETRIEVED CONTEXTS ---")
        for i, h in enumerate(hits, start=1):
            print(f"[{i}] dist={h['distance']:.4f} file={h['file']} chunk={h['chunk']}")
            print(f"    {h['text'][:200]}\n")

        try:
            answer = answer_with_ollama(q, hits)
        except Exception as e:
            print(f"\n[ERROR calling Ollama] {e}")
            continue

        print("\n--- ANSWER (OLLAMA) ---")
        print(answer)
        print("\n------------------------")
