import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from lab2.dto.models import Chunk, RetrievalMetrics


class Query(BaseModel):
    id: str
    lang: str
    book: str
    question: str
    evidence: list[str]
    source: str


class QuerySet(BaseModel):
    queries: list[Query]

    @classmethod
    def load(cls, path: Path) -> QuerySet:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def relevant(query: Query, chunks: list[Chunk]) -> set[int]:
    patterns = [re.compile(pattern, re.I) for pattern in query.evidence]
    return {
        index
        for index, chunk in enumerate(chunks)
        if chunk.book == query.book and all(p.search(re.sub(r"\s+", " ", chunk.text)) for p in patterns)
    }


@dataclass
class RankedQuery:
    query: Query
    ranking: list[int]
    relevant: set[int]

    def first_hit(self) -> int | None:
        return next((rank for rank, index in enumerate(self.ranking, start=1) if index in self.relevant), None)

    def ndcg(self, k: int) -> float:
        gains = [
            1.0 / math.log2(rank + 1) for rank, index in enumerate(self.ranking[:k], start=1) if index in self.relevant
        ]
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(self.relevant), k) + 1))
        return sum(gains) / ideal if ideal else 0.0


def rank(query_vectors: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    queries = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    chunks = chunk_vectors / np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    return np.argsort(-(queries @ chunks.T), axis=1)


def summarize(ranked: list[RankedQuery]) -> RetrievalMetrics:
    count = max(len(ranked), 1)
    hits = [item.first_hit() for item in ranked]
    return RetrievalMetrics(
        queries=len(ranked),
        hit_at_1=round(sum(1 for h in hits if h and h <= 1) / count, 4),
        hit_at_3=round(sum(1 for h in hits if h and h <= 3) / count, 4),
        hit_at_5=round(sum(1 for h in hits if h and h <= 5) / count, 4),
        mrr=round(sum(1.0 / h for h in hits if h) / count, 4),
        ndcg_at_10=round(sum(item.ndcg(10) for item in ranked) / count, 4),
    )
