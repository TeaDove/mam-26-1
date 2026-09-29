from dataclasses import dataclass

import httpx
import numpy as np
from pydantic import BaseModel


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingItem]


@dataclass
class EmbeddingClient:
    url: str
    model: str
    batch_size: int = 16

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=300) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(self.url, json={"model": self.model, "input": batch})
                response.raise_for_status()
                parsed = EmbeddingResponse.model_validate_json(response.content)
                vectors.extend(item.embedding for item in sorted(parsed.data, key=lambda item: item.index))
        return np.asarray(vectors, dtype=np.float32)
