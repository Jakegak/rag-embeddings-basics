import argparse
from typing import List, Dict

from rag_core import RAGState, build_rag_state, retrieve
from llm_backends import OpenAIBackend, OllamaBackend, LLMBackend

from pathlib import Path
import yaml


def load_retrieval_k(default: int = 4) -> int:
    path = Path("config.yaml")
    if path.exists():
        cfg = yaml.safe_load(path.read_text())
        return int(cfg.get("retrieval", {}).get("k", default))
    return default


def build_prompt(query: str, contexts: List[Dict]) -> str:
    context_block = ""
    for i, ctx in enumerate(contexts, start=1):
        context_block += (
            f"Context {i} (from {ctx['file']} chunk {ctx['chunk']}):\n"
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


def make_backend(kind: str) -> LLMBackend:
    kind = kind.lower()
    if kind == "openai":
        return OpenAIBackend()
    elif kind == "ollama":
        return OllamaBackend()
    else:
        raise ValueError(f"Unsupported llm-backend: {kind}. Use 'openai' or 'ollama'.")


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG CLI (OpenAI / Ollama).")
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="openai",
        help="Which LLM backend to use: 'openai' or 'ollama'.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=load_retrieval_k(),
        help="Number of chunks to retrieve.",
    )

    args = parser.parse_args()

    backend = make_backend(args.llm_backend)
    state: RAGState = build_rag_state()

    print(f"Using LLM backend: {args.llm_backend}")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        hits = retrieve(state, q, k=args.k)

        print("\n--- RETRIEVED CONTEXTS ---")
        for i, h in enumerate(hits, start=1):
            print(f"[{i}] dist={h['distance']:.4f} file={h['file']} chunk={h['chunk']}")
            print(f"    {h['text'][:200]}\n")

        prompt = build_prompt(q, hits)
        answer = backend.generate(prompt)

        print("\n--- ANSWER ---")
        print(answer)
        print("\n--------------------------")


if __name__ == "__main__":
    main()
