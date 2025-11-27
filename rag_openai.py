import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# --------- Config ---------
DATA_DIR = Path("data")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model
OPENAI_MODEL = "gpt-4o-mini"
# --------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(EMBED_MODEL_NAME)

documents: list[str] = []
metadatas: list[dict] = []  # {file, chunk}


def load_files():
    for path in DATA_DIR.glob("**/*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # simple paragraph-based chunking
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"file": str(path), "chunk": i})


def build_index():
    load_files()
    if not documents:
        raise ValueError("No documents found in the data directory.")

    print(f"Loaded {len(documents)} chunks from {len(set(m['file'] for m in metadatas))} files.")

    embeddings = model.encode(documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} chunks.")
    return index, embeddings


index, embeddings = build_index()


def retrieve(query: str, k: int = 4):
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = metadatas[idx]
        results.append(
            {
                "distance": float(dist),
                "file": meta["file"],
                "chunk": meta["chunk"],
                "text": documents[idx],  # 🔹 use 'text' key consistently
            }
        )
    return results


def build_prompt(query: str, contexts: list[dict]) -> str:
    contexts_block = ""
    for i, ctx in enumerate(contexts, start=1):
        contexts_block += (
            f"Context {i} (from {ctx['file']} chunk {ctx['chunk']}):\n"
            f"{ctx['text']}\n\n"  # 🔹 use 'text'
        )

    prompt = (
        "You are a helpful assistant answering questions based ONLY on the given context.\n"
        "If the answer is not in the context, say you don't know and do NOT hallucinate.\n\n"
        f"Context:\n{contexts_block}"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt


def answer_with_openai(query: str, contexts: list[dict]) -> str:
    prompt = build_prompt(query, contexts)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a grounded RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        hits = retrieve(q, k=4)
        print("\n--- RETRIEVED CONTEXTS ---")
        for i, h in enumerate(hits, start=1):
            print(f"[{i}] dist={h['distance']:.4f} file={h['file']} chunk={h['chunk']}")
            print(f"    {h['text'][:200]}\n")  # 🔹 matches 'text' key

        answer = answer_with_openai(q, hits)
        print("\n--- ANSWER ---")
        print(answer)
        print("\n--------------------------")
