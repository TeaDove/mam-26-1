import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from lab2.dto.models import VectorizationMetrics


@dataclass
class VectorDocuments:
    ids: list[str]
    langs: list[str]
    windows: list[list[str]]
    dense: np.ndarray


def word_windows(words: list[str], size: int) -> list[list[str]]:
    return [words[start : start + size] for start in range(0, len(words), size) if words[start : start + size]]


def _tfidf(windows: list[list[str]]) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(analyzer=lambda words: [w.casefold() for w in words])
    return vectorizer, vectorizer.fit_transform(windows)


def save_tfidf(docs: VectorDocuments, prefix: Path) -> None:
    vectorizer, matrix = _tfidf(docs.windows)
    sparse.save_npz(prefix.with_suffix(".npz"), matrix)
    prefix.with_suffix(".vocab.json").write_text(
        json.dumps(sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get), ensure_ascii=False),
        encoding="utf-8",
    )


def vectorization_metrics(docs: VectorDocuments) -> VectorizationMetrics:
    _, matrix = _tfidf(docs.windows)
    cells = matrix.shape[0] * matrix.shape[1]
    normalized = docs.dense / np.linalg.norm(docs.dense, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    count = len(docs.ids)
    off_diagonal = similarity[~np.eye(count, dtype=bool)]
    ru = [i for i, lang in enumerate(docs.langs) if lang == "ru"]
    en = [i for i, lang in enumerate(docs.langs) if lang == "en"]
    cross_mean = cross_best = None
    if ru and en:
        block = similarity[np.ix_(ru, en)]
        cross_mean = round(float(block.mean()), 4)
        cross_best = round(float(block.max(axis=1).mean()), 4)
    return VectorizationMetrics(
        documents=count,
        tfidf_windows=matrix.shape[0],
        tfidf_vocabulary=matrix.shape[1],
        tfidf_sparsity=round(1 - matrix.nnz / max(cells, 1), 4),
        tfidf_mean_nonzero_per_doc=round(matrix.nnz / max(matrix.shape[0], 1), 1),
        dense_dimensions=int(docs.dense.shape[1]),
        dense_mean_pairwise_cosine=round(float(off_diagonal.mean()), 4) if count > 1 else 0.0,
        cross_language_mean_cosine=cross_mean,
        cross_language_best_match_cosine=cross_best,
    )
