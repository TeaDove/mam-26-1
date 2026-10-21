from dataclasses import dataclass

import numpy as np

from lab3.graph import KnowledgeGraph
from lab3.models import QueryGraph


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    seed_vertices: int
    context_vertices: int
    relation_bonus: float
    two_hop_bonus: float


@dataclass(frozen=True)
class Retrieval:
    ranking: list[str]
    seeds: list[str]
    context: list[str]


def query_edges(query: QueryGraph) -> np.ndarray:
    index = {entity.name: i for i, entity in enumerate(query.entities)}
    pairs = [(index[r.source], index[r.target]) for r in query.relations if r.source in index and r.target in index]
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(pairs, dtype=np.int64).T


def relation_bonus(
    kg: KnowledgeGraph, candidates: list[np.ndarray], pairs: np.ndarray, config: RetrievalConfig
) -> np.ndarray:
    bonus = np.zeros(len(kg.nodes))
    for a, b in pairs.T:
        for u in candidates[a]:
            for w in candidates[b]:
                if u == w:
                    continue
                left, right = kg.nodes[u], kg.nodes[w]
                if kg.graph.has_edge(left, right):
                    gain = config.relation_bonus
                elif set(kg.graph[left]) & set(kg.graph[right]):
                    gain = config.two_hop_bonus
                else:
                    continue
                bonus[u] = max(bonus[u], gain)
                bonus[w] = max(bonus[w], gain)
    return bonus


def expand(kg: KnowledgeGraph, seeds: list[str], score: dict[str, float], limit: int) -> list[str]:
    context = list(dict.fromkeys(seeds))[:limit]
    frontier = seeds
    while len(context) < limit and frontier:
        chosen = set(context)
        neighbours = {n for node in frontier for n in kg.graph[node] if n not in chosen}
        ordered = sorted(neighbours, key=lambda n: (-score[n], n))
        context.extend(ordered[: limit - len(context)])
        frontier = ordered
    return context


def source_units(kg: KnowledgeGraph, seeds: list[str], context: list[str], limit: int) -> list[str]:
    candidates = list(dict.fromkeys(unit for seed in seeds for unit in kg.units.get(seed, []) if unit in kg.unit_text))
    support = {unit: sum(unit in kg.units.get(node, []) for node in context) for unit in candidates}
    return sorted(candidates, key=lambda unit: -support[unit])[:limit]


def combine(blocks: list[tuple[float, np.ndarray]]) -> np.ndarray:
    return np.hstack([np.sqrt(weight) * vectors for weight, vectors in blocks]).astype(np.float32)


def retrieve(
    kg: KnowledgeGraph,
    vertex_vectors: np.ndarray,
    query: QueryGraph,
    query_vectors: np.ndarray,
    config: RetrievalConfig,
) -> Retrieval:
    similarity = query_vectors @ vertex_vectors.T
    candidates = [np.argsort(-row, kind="stable")[: config.top_k] for row in similarity]
    total = similarity.max(axis=0) + relation_bonus(kg, candidates, query_edges(query), config)
    order = np.argsort(-total, kind="stable")
    ranking = [kg.nodes[i] for i in order]
    score = {kg.nodes[i]: float(total[i]) for i in range(len(kg.nodes))}
    seeds = ranking[: config.seed_vertices]
    return Retrieval(ranking=ranking, seeds=seeds, context=expand(kg, seeds, score, config.context_vertices))
