import re
import statistics
from dataclasses import dataclass

import networkx as nx
import numpy as np

from lab2.dto.models import (
    Completeness,
    Consistency,
    ConsistencyVerdict,
    Coreference,
    Coverage,
    GraphReport,
    GraphStructure,
    InputStats,
    Integrity,
    NoiseReport,
    Traversal,
)
from lab2.service.graph_criteria import (
    IntegrityResult,
    duplicate_groups,
    embedding_duplicates,
    formula_atoms,
    formulas_and_tables,
    integrity,
    lemma_key,
    lift_pronouns,
    noise_labels,
    rule_counts,
    sample_nodes,
    table_atoms,
    window,
)
from lab2.service.graph_eval import EvaluationConfig, GraphBundle, coverage, structure_metrics, traversal
from lab2.service.judge import check_consistency
from lab2.supplier.embeddings import EmbeddingClient
from lab2.supplier.llm import LlmClient


@dataclass
class ComparisonContext:
    config: EvaluationConfig
    client: LlmClient
    embedder: EmbeddingClient
    terms: list[str]
    term_vectors: np.ndarray
    concept_vectors: np.ndarray
    vocabulary: frozenset[str]
    allowed: frozenset[str]

    def evaluate(self, bundle: GraphBundle, source: str) -> tuple[GraphReport, list[ConsistencyVerdict]]:
        graph = bundle.graph
        names = [str(n) for n in graph.nodes]
        vectors = self.embedder.embed(names)
        noise = {n: noise_labels(n, self.vocabulary, self.allowed, graph.nodes[n]["type"]) for n in names}
        noisy = [n for n, labels in noise.items() if labels]
        domain = self.coverage(graph, vectors)
        verdicts = self.consistency_verdicts(bundle, list(domain.resolved.values()))
        report = GraphReport(
            input_stats=input_stats(bundle, len(names), len(noisy)),
            structure=GraphStructure.model_validate(structure_metrics(graph)),
            completeness=self.completeness(names, vectors),
            noise=noise_report(names, noise),
            coreference=coreference(graph, names, vectors, self.config.duplicate_similarity),
            integrity=integrity_report(graph, source),
            consistency=consistency_summary(verdicts),
            coverage=domain,
            traversal=Traversal.model_validate(traversal(graph, self.config, {n: "noise" for n in noisy})),
        )
        return report, verdicts

    def completeness(self, names: list[str], vectors: np.ndarray) -> Completeness:
        similarity = cosine(self.term_vectors, vectors)
        keys = [f" {lemma_key(n)} " for n in names]
        exact = [any(f" {lemma_key(term)} " in key for key in keys) for term in self.terms]
        semantic = [bool(e or similarity[i].max() >= self.config.term_similarity) for i, e in enumerate(exact)]
        total = max(len(self.terms), 1)
        return Completeness(
            reference_terms=len(self.terms),
            covered_exact=sum(exact),
            covered_exact_share=round(sum(exact) / total, 4),
            covered_semantic=sum(semantic),
            covered_semantic_share=round(sum(semantic) / total, 4),
            missing_examples=[t for t, s in zip(self.terms, semantic, strict=True) if not s][:25],
        )

    def coverage(self, graph: nx.Graph, vectors: np.ndarray) -> Coverage:
        concepts = list(self.config.concepts)
        similarity = cosine(self.concept_vectors, vectors).max(axis=1)
        found = [c for c, s in zip(concepts, similarity, strict=True) if s >= self.config.term_similarity]
        return Coverage(
            **coverage(graph, self.config),
            vector_found=len(found),
            vector_coverage=round(len(found) / max(len(concepts), 1), 4),
        )

    def consistency_verdicts(self, bundle: GraphBundle, must: list[str]) -> list[ConsistencyVerdict]:
        graph = bundle.graph
        entity_units = {
            row.title: list(row.text_unit_ids)
            for row in bundle.artifacts.entities.itertuples()
            if len(row.text_unit_ids)
        }
        unit_text = dict(zip(bundle.artifacts.text_units["id"], bundle.artifacts.text_units["text"], strict=True))
        return [
            check_consistency(
                self.client,
                graph,
                node,
                window(graph, node),
                source_fragment(node, entity_units.get(node, []), unit_text),
            )
            for node in sample_nodes(graph, must, self.config.consistency_sample)
        ]


def input_stats(bundle: GraphBundle, vertices: int, noisy: int) -> InputStats:
    tokens = int(bundle.artifacts.text_units["n_tokens"].sum())
    return InputStats(
        text_units=len(bundle.artifacts.text_units),
        input_tokens=tokens,
        vertices_per_1k_tokens=round(vertices / tokens * 1000, 2),
        noise_vertices_per_1k_tokens=round(noisy / tokens * 1000, 2),
    )


def noise_report(names: list[str], noise: dict[str, list[str]]) -> NoiseReport:
    noisy = [n for n, labels in noise.items() if labels]
    return NoiseReport(
        vertices=len(names),
        noise_vertices=len(noisy),
        noise_share=round(len(noisy) / max(len(names), 1), 4),
        by_rule=rule_counts(noise),
        examples=sorted(noisy, key=str)[:40],
    )


def coreference(graph: nx.Graph, names: list[str], vectors: np.ndarray, threshold: float) -> Coreference:
    groups = duplicate_groups(names)
    pairs = embedding_duplicates(names, vectors, threshold)
    cross = [p for p in pairs if {graph.nodes[p[0]]["lang"], graph.nodes[p[1]]["lang"]} == {"en", "ru"}]
    lifted, subtree = lift_pronouns(graph)
    return Coreference(
        lemma_duplicate_groups=len(groups),
        lemma_duplicate_vertices=sum(len(g) for g in groups),
        embedding_duplicate_pairs=len(pairs),
        cross_language_pairs=len(cross),
        pronoun_vertices_lifted=lifted,
        pronoun_subtree_vertices=subtree,
        lemma_examples=[" = ".join(g) for g in groups[:15]],
        embedding_examples=[f"{a} = {b} ({s})" for a, b, s in pairs[:15]],
    )


def integrity_report(graph: nx.Graph, source: str) -> Integrity:
    formulas, tables = formulas_and_tables(source)
    formula_result = integrity(graph, [formula_atoms(f) for f in formulas])
    table_result = integrity(graph, [table_atoms(cells) for cells in tables])
    return Integrity(
        formulas=len(formulas),
        formulas_evaluated=evaluated(formula_result),
        formulas_in_vertex=formula_result.in_vertex,
        formulas_in_window=formula_result.in_window,
        formulas_broken=formula_result.broken,
        formula_mean_window_share=round(statistics.fmean(formula_result.best_share or [0.0]), 3),
        tables=len(tables),
        tables_evaluated=evaluated(table_result),
        tables_in_vertex=table_result.in_vertex,
        tables_in_window=table_result.in_window,
        tables_broken=table_result.broken,
        table_best_window_share=table_result.best_share,
    )


def evaluated(result: IntegrityResult) -> int:
    return result.in_vertex + result.in_window + result.broken


def consistency_summary(verdicts: list[ConsistencyVerdict]) -> Consistency:
    total = max(len(verdicts), 1)
    return Consistency(
        checked_vertices=len(verdicts),
        fully_defined_share=round(sum(v.definition == "full" for v in verdicts) / total, 4),
        partially_defined_share=round(sum(v.definition == "partial" for v in verdicts) / total, 4),
        contradictions_total=sum(v.contradictions for v in verdicts),
        vertices_with_contradictions_share=round(sum(v.contradictions > 0 for v in verdicts) / total, 4),
    )


def source_fragment(node: str, units: list[str], unit_text: dict[str, str]) -> str:
    best = max(units, key=lambda unit: mentions(node, str(unit_text.get(unit, ""))), default="")
    return str(unit_text.get(best, ""))


def mentions(name: str, text: str) -> int:
    total = 0
    for word in re.findall(r"[^\W_]+", name):
        if len(word) < 4:
            total += len(re.findall(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])", text))
        else:
            stem = word[: max(4, len(word) - 2)]
            total += len(re.findall(rf"(?<![^\W_]){re.escape(stem)}", text, re.I))
    return total


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = left / np.linalg.norm(left, axis=1, keepdims=True)
    b = right / np.linalg.norm(right, axis=1, keepdims=True)
    return a @ b.T
