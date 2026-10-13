from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from lab2.dto.models import GraphComparison, GraphReport, VariantMetric

CLEAR_NOISE_FACTOR = 2.0
CLEAR_SHARE_POINTS = 0.03
CLEAR_COUNT_RELATIVE = 0.1


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: Literal["share", "count"]
    value: Callable[[GraphReport], float]


METRICS = (
    MetricSpec("text units", "count", lambda r: r.input_stats.text_units),
    MetricSpec("input tokens", "count", lambda r: r.input_stats.input_tokens),
    MetricSpec("vertices per 1k tokens", "count", lambda r: r.input_stats.vertices_per_1k_tokens),
    MetricSpec("vertices", "count", lambda r: r.structure.nodes),
    MetricSpec("edges", "count", lambda r: r.structure.edges),
    MetricSpec("connected components", "count", lambda r: r.structure.connected_components),
    MetricSpec("largest component share", "share", lambda r: r.structure.largest_component_share),
    MetricSpec("isolated vertices", "count", lambda r: r.structure.isolated_nodes),
    MetricSpec("RU vertices in largest component", "count", lambda r: r.structure.ru_nodes_in_largest_component),
    MetricSpec("cycles per 100 vertices", "count", lambda r: r.cycles.cycles_per_100_vertices),
    MetricSpec("cross-book cycles", "count", lambda r: r.cycles.cross_book_cycles),
    MetricSpec("completeness (semantic)", "share", lambda r: r.completeness.covered_semantic_share),
    MetricSpec("domain coverage", "share", lambda r: r.coverage.domain_coverage),
    MetricSpec("connected concept pairs", "count", lambda r: r.traversal.summary.pairs_connected),
    MetricSpec("RU book share in BFS-2", "share", lambda r: r.traversal.summary.bfs2_ru_book_share),
    MetricSpec("noise share", "share", lambda r: r.noise.noise_share),
    MetricSpec("formula vertices", "count", lambda r: r.formulas.vertices),
    MetricSpec("translation duplicates", "count", lambda r: r.coreference.translation_pairs),
    MetricSpec("same-language duplicates", "count", lambda r: r.coreference.same_language_pairs),
    MetricSpec("word-form duplicate vertices", "count", lambda r: r.coreference.lemma_duplicate_vertices),
    MetricSpec("formula window share", "share", lambda r: r.integrity.formula_mean_window_share),
    MetricSpec("vertices with contradictions", "share", lambda r: r.consistency.vertices_with_contradictions_share),
)


def is_clear(spec: MetricSpec, baseline: float, delta: float, noise: float) -> bool:
    floor = CLEAR_SHARE_POINTS if spec.kind == "share" else CLEAR_COUNT_RELATIVE * abs(baseline)
    return abs(delta) > CLEAR_NOISE_FACTOR * noise and abs(delta) >= floor


def compare_variants(
    baseline: GraphComparison, rerun: GraphComparison, overlap: GraphComparison
) -> list[VariantMetric]:
    rows: list[VariantMetric] = []
    for spec in METRICS:
        base = float(spec.value(baseline.clean))
        noise = max(
            abs(spec.value(rerun.clean) - base),
            abs(spec.value(rerun.dirty) - spec.value(baseline.dirty)),
            abs(spec.value(overlap.dirty) - spec.value(baseline.dirty)),
        )
        delta = spec.value(overlap.clean) - base
        rows.append(
            VariantMetric(
                name=spec.name,
                kind=spec.kind,
                baseline=base,
                rerun=float(spec.value(rerun.clean)),
                overlap=float(spec.value(overlap.clean)),
                noise=round(float(noise), 4),
                delta=round(float(delta), 4),
                clear=is_clear(spec, base, delta, noise),
            )
        )
    return rows
