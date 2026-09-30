import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml
from pydantic import BaseModel


class EvaluationConfig(BaseModel):
    duplicate_similarity: float
    term_similarity: float
    terms_en: int
    terms_ru: int
    consistency_sample: int
    bfs_depths: list[int]
    concepts: dict[str, list[str]]
    seeds: list[str]
    pairs: list[tuple[str, str]]

    @classmethod
    def load(cls, path: Path) -> EvaluationConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class GraphArtifacts(BaseModel, arbitrary_types_allowed=True):
    entities: pd.DataFrame
    relationships: pd.DataFrame
    text_units: pd.DataFrame
    documents: pd.DataFrame


@dataclass
class GraphBundle:
    graph: nx.Graph
    artifacts: GraphArtifacts


def load_graph(output: Path) -> GraphBundle:
    artifacts = GraphArtifacts(
        entities=pd.read_parquet(output / "entities.parquet"),
        relationships=pd.read_parquet(output / "relationships.parquet"),
        text_units=pd.read_parquet(output / "text_units.parquet"),
        documents=pd.read_parquet(output / "documents.parquet"),
    )
    titles = dict(zip(artifacts.documents["id"], artifacts.documents["title"], strict=True))
    unit_book = {row.id: Path(titles[row.document_id]).stem.split("_")[0] for row in artifacts.text_units.itertuples()}
    graph = nx.Graph()
    for row in artifacts.entities.itertuples():
        books = sorted({unit_book[unit] for unit in row.text_unit_ids if unit in unit_book})
        graph.add_node(
            row.title,
            type=str(row.type or "").lower(),
            description=str(row.description or ""),
            frequency=int(row.frequency),
            books=",".join(books),
            lang=_lang_of(books),
        )
    for row in artifacts.relationships.itertuples():
        for endpoint in (row.source, row.target):
            if endpoint not in graph:
                graph.add_node(endpoint, type="", description="", frequency=0, books="", lang="unknown")
        if graph.has_edge(row.source, row.target):
            edge = graph[row.source][row.target]
            edge["weight"] += float(row.weight)
            edge["description"] = f"{edge['description']}\n{row.description}"
            edge["merged"] = edge.get("merged", 1) + 1
        else:
            graph.add_edge(row.source, row.target, weight=float(row.weight), description=str(row.description))
    return GraphBundle(graph=graph, artifacts=artifacts)


def _lang_of(books: list[str]) -> str:
    langs = {"ru" if book == "stat3" else "en" for book in books}
    if len(langs) == 2:
        return "both"
    return next(iter(langs), "unknown")


def structure_metrics(graph: nx.Graph) -> dict[str, object]:
    degrees = [degree for _, degree in graph.degree()]
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0]) if components else nx.Graph()
    types = Counter(data["type"] or "unknown" for _, data in graph.nodes(data=True))
    pair_types = Counter(
        " — ".join(sorted((graph.nodes[a]["type"] or "unknown", graph.nodes[b]["type"] or "unknown")))
        for a, b in graph.edges()
    )
    langs = Counter(data["lang"] for _, data in graph.nodes(data=True))
    cross = sum(1 for a, b in graph.edges() if {graph.nodes[a]["lang"], graph.nodes[b]["lang"]} == {"en", "ru"})
    touches_both = sum(1 for a, b in graph.edges() if "both" in (graph.nodes[a]["lang"], graph.nodes[b]["lang"]))
    mixed_components = sum(
        1
        for c in components
        if {"en", "ru"} <= {graph.nodes[n]["lang"] for n in c} or any(graph.nodes[n]["lang"] == "both" for n in c)
    )
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 5),
        "connected_components": len(components),
        "largest_component_nodes": largest.number_of_nodes(),
        "largest_component_share": round(largest.number_of_nodes() / max(graph.number_of_nodes(), 1), 4),
        "isolated_nodes": sum(1 for d in degrees if d == 0),
        "bridges": sum(1 for _ in nx.bridges(graph)),
        "articulation_points": sum(1 for _ in nx.articulation_points(graph)),
        "cyclomatic_number": graph.number_of_edges() - graph.number_of_nodes() + len(components),
        "cycle_basis": len(nx.cycle_basis(graph)),
        "average_clustering": round(nx.average_clustering(graph), 4),
        "degree_mean": round(statistics.fmean(degrees), 3),
        "degree_median": float(statistics.median(degrees)),
        "degree_max": max(degrees),
        "leaves": sum(1 for d in degrees if d == 1),
        "lcc_average_shortest_path": round(nx.average_shortest_path_length(largest), 3)
        if largest.number_of_nodes() > 1
        else 0.0,
        "lcc_diameter": nx.diameter(largest) if largest.number_of_nodes() > 1 else 0,
        "nodes_en": langs.get("en", 0),
        "nodes_ru": langs.get("ru", 0),
        "nodes_both_languages": langs.get("both", 0),
        "edges_en_ru": cross,
        "edges_touching_bilingual_nodes": touches_both,
        "components_mixing_languages": mixed_components,
        "nodes_by_type": dict(types.most_common()),
        "edges_by_type_pair": dict(pair_types.most_common()),
        "top_hubs": [f"{node} ({degree})" for node, degree in sorted(graph.degree(), key=lambda x: -x[1])[:12]],
    }


def resolve(graph: nx.Graph, aliases: list[str]) -> str | None:
    index: dict[str, list[str]] = {}
    for node in graph.nodes:
        key = str(node).casefold()
        index.setdefault(key, []).append(node)
        head = re.sub(r"\s*\(.*$", "", key).strip()
        if head != key:
            index.setdefault(head, []).append(node)
    candidates = [node for alias in aliases for node in index.get(alias.casefold(), [])]
    if not candidates:
        return None
    return max(candidates, key=lambda node: graph.degree(node))


def coverage(graph: nx.Graph, config: EvaluationConfig) -> dict[str, object]:
    resolved = {concept: resolve(graph, aliases) for concept, aliases in config.concepts.items()}
    found = {concept: node for concept, node in resolved.items() if node}
    return {
        "concepts_total": len(config.concepts),
        "concepts_found": len(found),
        "domain_coverage": round(len(found) / max(len(config.concepts), 1), 4),
        "missing": sorted(set(config.concepts) - set(found)),
        "resolved": found,
    }


def traversal(graph: nx.Graph, config: EvaluationConfig, labels: dict[str, str]) -> dict[str, object]:
    concept_nodes = {n for n in (resolve(graph, a) for a in config.concepts.values()) if n}
    seeds: dict[str, object] = {}
    for concept in config.seeds:
        node = resolve(graph, config.concepts[concept])
        if node is None:
            seeds[concept] = None
            continue
        entry: dict[str, object] = {"node": node}
        lengths = nx.single_source_shortest_path_length(graph, node)
        for depth in config.bfs_depths:
            reached = {n for n, d in lengths.items() if 0 < d <= depth}
            entry[f"bfs{depth}_reached"] = len(reached)
            entry[f"bfs{depth}_noise"] = sum(1 for n in reached if labels.get(n) == "noise")
            entry[f"bfs{depth}_concepts"] = len(reached & concept_nodes)
            entry[f"bfs{depth}_other_language"] = sum(
                1 for n in reached if graph.nodes[n]["lang"] not in (graph.nodes[node]["lang"], "both")
            )
        tree = nx.dfs_tree(graph, node)
        depth_map = nx.single_source_shortest_path_length(tree, node)
        entry["dfs_reached"] = tree.number_of_nodes() - 1
        entry["dfs_max_depth"] = max(depth_map.values())
        seeds[concept] = entry
    pairs: list[dict[str, object]] = []
    for source, target in config.pairs:
        a, b = resolve(graph, config.concepts[source]), resolve(graph, config.concepts[target])
        row: dict[str, object] = {"source": source, "target": target, "source_node": a, "target_node": b}
        if a and b and nx.has_path(graph, a, b):
            started = time.perf_counter()
            path = nx.shortest_path(graph, a, b)
            row["shortest_ms"] = round((time.perf_counter() - started) * 1000, 3)
            started = time.perf_counter()
            bidirectional = nx.bidirectional_shortest_path(graph, a, b)
            row["bidirectional_ms"] = round((time.perf_counter() - started) * 1000, 3)
            row["length"] = len(path) - 1
            row["path"] = path
            row["bidirectional_length"] = len(bidirectional) - 1
        else:
            row["length"] = None
        pairs.append(row)
    found_pairs = [p for p in pairs if p["length"] is not None]
    resolved_seeds = [s for s in seeds.values() if isinstance(s, dict)]
    summary: dict[str, object] = {
        "seeds_resolved": len(resolved_seeds),
        "pairs_total": len(pairs),
        "pairs_connected": len(found_pairs),
        "pairs_mean_length": round(statistics.fmean([p["length"] for p in found_pairs]), 3) if found_pairs else None,
    }
    for depth in config.bfs_depths:
        reached = sum(s[f"bfs{depth}_reached"] for s in resolved_seeds)
        noise = sum(s[f"bfs{depth}_noise"] for s in resolved_seeds)
        summary[f"bfs{depth}_mean_reached"] = round(reached / max(len(resolved_seeds), 1), 2)
        summary[f"bfs{depth}_mean_concepts"] = round(
            sum(s[f"bfs{depth}_concepts"] for s in resolved_seeds) / max(len(resolved_seeds), 1), 2
        )
        summary[f"bfs{depth}_noise_share"] = round(noise / max(reached, 1), 4)
        summary[f"bfs{depth}_other_language_share"] = round(
            sum(s[f"bfs{depth}_other_language"] for s in resolved_seeds) / max(reached, 1), 4
        )
    summary["dfs_mean_reached"] = (
        round(statistics.fmean([s["dfs_reached"] for s in resolved_seeds]), 2) if resolved_seeds else None
    )
    return {"summary": summary, "seeds": seeds, "pairs": pairs}
