from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

from rag_core import RAGState, build_rag_state, retrieve
from llm_backends import OpenAIBackend, OllamaBackend, LLMBackend


EVAL_FILE = Path("eval_dataset.jsonl")


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


def load_eval_data(path: Path = EVAL_FILE) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")
    examples: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def make_backend(kind: str = "openai") -> LLMBackend:
    kind = kind.lower()
    if kind == "openai":
        return OpenAIBackend()
    elif kind == "ollama":
        return OllamaBackend()
    else:
        raise ValueError(f"Unsupported backend: {kind}")


def evaluate(backend_name: str = "ollama", k: int = 4):
    print(f"Loading RAG state...")
    state: RAGState = build_rag_state()
    backend = make_backend(backend_name)
    examples = load_eval_data()

    total = len(examples)
    retrieval_hits = 0
    keyword_hits = 0

    per_example_results = []

    for ex in examples:
        query = ex["query"]
        expected_keywords: List[str] = ex.get("expected_answer_keywords", [])
        expected_files: List[str] = ex.get("expected_files", [])

        hits = retrieve(state, query, k=k)
        retrieved_files = {h["file"] for h in hits}

        # Retrieval metric: did we retrieve any of the expected files?
        retrieved_correct = any(f in retrieved_files for f in expected_files)
        if retrieved_correct:
            retrieval_hits += 1

        prompt = build_prompt(query, hits)
        answer = backend.generate(prompt)

        # Simple keyword-based correctness check
        normalized_answer = answer.lower()
        matched_keywords = [
            kw for kw in expected_keywords if kw.lower() in normalized_answer
        ]
        if matched_keywords:
            keyword_hits += 1

        per_example_results.append(
            {
                "query": query,
                "retrieved_correct": retrieved_correct,
                "retrieved_files": list(retrieved_files),
                "expected_files": expected_files,
                "matched_keywords": matched_keywords,
                "expected_keywords": expected_keywords,
            }
        )

    print("\n=== RAG EVALUATION REPORT ===")
    print(f"Backend: {backend_name}")
    print(f"Total examples: {total}")
    print(f"Retrieval hit@{k}: {retrieval_hits}/{total} = {retrieval_hits / total:.2f}")
    print(f"Answer keyword hit: {keyword_hits}/{total} = {keyword_hits / total:.2f}")
    print("\n--- Per-example details ---")
    for res in per_example_results:
        print(f"\nQuery: {res['query']}")
        print(f"  Retrieved correct file? {res['retrieved_correct']}")
        print(f"  Retrieved files: {res['retrieved_files']}")
        print(f"  Expected files:  {res['expected_files']}")
        print(f"  Matched keywords: {res['matched_keywords']}")
        print(f"  Expected keywords: {res['expected_keywords']}")


if __name__ == "__main__":
    # You can tweak backend or k here if you like
    evaluate(backend_name="openai", k=4)
