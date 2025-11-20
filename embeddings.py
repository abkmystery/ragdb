"""
embeddings.py

Small helpers for calling external embedding providers such as
OpenAI and NVIDIA. These are not required for basic TF–IDF search,
but you keep them here so you can experiment with cloud embeddings on
top of the same SQLite file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


class EmbeddingError(RuntimeError):
    pass


@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"

    def embed_texts(self, texts: Sequence[str], api_key: str | None = None) -> List[List[float]]:
        from openai import OpenAI  # type: ignore[import]

        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in resp.data]  # type: ignore[attr-defined]


@dataclass
class NvidiaEmbedder:
    endpoint: str
    model: str

    def embed_texts(self, texts: Sequence[str], api_key: str) -> List[List[float]]:
        import requests  # type: ignore[import]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": list(texts)}
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise EmbeddingError(
                f"NVIDIA embeddings request failed: {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        items = data.get("data") or []
        return [item["embedding"] for item in items]
