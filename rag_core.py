from  __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import yaml

# --------- CONFIG ------------
DATA_DIR = Path("data")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# -----------------------------

def load_config_from_yaml(path: Path = Path("config.yaml")) -> RAGConfig:
    if path.exists():
        raw = yaml.safe_load(path.read_text())
        data_dir = Path(raw.get("data_dir", "data"))
        embed_model_name = raw.get("embed_model_name", EMBED_MODEL_NAME)
        return RAGConfig(data_dir=data_dir, embed_model_name=embed_model_name)
    return RAGConfig()

@dataclass
class RAGConfig:
    data_dir: Path = DATA_DIR
    embed_model_name: str = EMBED_MODEL_NAME

@dataclass
class RAGState:
    config: RAGConfig
    model: SentenceTransformer
    index: faiss.Index
    documents: List[str]
    metadatas: List[Dict]

def load_files(data_dir: Path) -> Tuple[List[str], List[Dict]]:
    documents = []
    metadatas = []
    for path in data_dir.glob("**/*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # simple paragraph-based chunking
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"file": str(path), "chunk": i})

    return documents, metadatas

def build_index(documents: List[str], embed_model: str) -> Tuple[SentenceTransformer, faiss.Index]:
    if not documents:
        raise ValueError("No documents found in the data directory.")
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index

def build_rag_state(config: Optional[RAGConfig] = None) -> RAGState:
    if config is None:
        config = load_config_from_yaml()

    documents, metadatas = load_files(config.data_dir)
    if not documents:
        raise ValueError("No documents loaded from the data directory.")
    
    print(f"Loaded {len(documents)} chunks from {len(set(m['file'] for m in metadatas))} files.")

    model, index = build_index(documents, config.embed_model_name)
    print(f"Indexed {index.ntotal} chunks.")

    return RAGState(
        config=config,
        model=model,
        index=index,
        documents=documents,
        metadatas=metadatas
    )

def retrieve(
    state: RAGState,
    query: str,
    k: int = 4,
) -> List[Dict]:
    query_emb = state.model.encode([query], convert_to_numpy=True)
    distances, indices = state.index.search(query_emb, k)

    results: List[Dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = state.metadatas[idx]
        results.append(
            {
                "distance": float(dist),
                "file": meta["file"],
                "chunk": meta["chunk"],
                "text": state.documents[idx],
            }
        )
    return results