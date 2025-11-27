from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests
from openai import OpenAI
from dotenv import load_dotenv

from pathlib import Path
import yaml


load_dotenv()

def _load_yaml(path: Path = Path("config.yaml")) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text())
    return {}


class LLMBackend(ABC):
    """Abstract interface for any LLM backend."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response given a prompt."""
        ...


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        cfg = _load_yaml()
        openai_cfg = cfg.get("llm_backends", {}).get("openai", {})
        model = model or openai_cfg.get("model", "gpt-4o-mini")

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a grounded RAG assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()


class OllamaBackend(LLMBackend):
    def __init__(self, model: str | None = None, url: str | None = None):
        cfg = _load_yaml()
        ollama_cfg = cfg.get("llm_backends", {}).get("ollama", {})

        # Priority: explicit arg > env var > config > hard-coded default
        self.model = (
            model
            or os.getenv("OLLAMA_MODEL")
            or ollama_cfg.get("model")
            or "llama3"
        )
        self.url = (
            url
            or os.getenv("OLLAMA_URL")
            or ollama_cfg.get("url")
            or "http://localhost:11434/api/chat"
        )

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a grounded RAG assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = requests.post(self.url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
