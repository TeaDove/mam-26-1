import json
import statistics
from dataclasses import dataclass

import networkx as nx
import numpy as np

from lab2.service.graph_eval import GraphBundle
from lab2.supplier.llm import LlmClient

NODE_SYSTEM = (
    "You are an expert in physical metallurgy and steel rolling. You audit nodes of a knowledge graph built "
    "from two texts on controlled rolling of steel (an English review and a Russian article). "
    "Label every node strictly: 'core' = a correctly named, meaningful domain concept, material, process, "
    "parameter, property, phase or equipment, or a researcher cited for a scientific result; 'peripheral' = "
    "correct but generic or marginal (a bare symbol, a generic word like STEEL or WATER, a figure-specific value); "
    "'noise' = any of: Latin transliteration of Russian words, misspelled, "
    "OCR-broken, hyphen-broken or glued names, words unrelated "
    "to the domain, formatting leftovers, bibliographic metadata (journals, publishers, report codes), the authors "
    "and affiliations of the source article itself, or fragments without domain meaning. When a name is visibly "
    "corrupted, label it noise even if the concept behind it is valid. "
    'Answer JSON: {"labels": [{"id": <int>, "label": "core"|"peripheral"|"noise"}]} covering every id.'
)
UNIT_SYSTEM = (
    "You are an expert in physical metallurgy and steel rolling. You evaluate how well a knowledge-graph "
    "extractor captured one text fragment. You get the fragment and the entities and relations extracted from it. "
    "Score strictly and consistently. Answer JSON with keys: "
    '"completeness" (1-5: are the key domain facts of the fragment represented by entities and relations), '
    '"missing_key_facts" (list of up to 3 short strings), '
    '"contradictions" (int: extracted statements that contradict or distort the fragment), '
    '"coreference_duplicates" (int: extra entities that denote the same thing as another extracted entity), '
    '"noise_entities" (int: entities that are garbage, OCR artifacts, formatting leftovers or bibliographic noise), '
    '"formula_table_integrity" (1-5, or null if the fragment has no formulas, equations, numeric tables or '
    "quantities: are formulas, numbers with units and table data carried into entities/descriptions correctly), "
    '"domain_coverage" (1-5: share of the fragment\'s domain concepts present as entities).'
)
DUPLICATE_SYSTEM = (
    "You check candidate duplicate nodes of a knowledge graph about controlled rolling of steel. The graph may mix "
    "English and Russian names. For every pair decide whether both nodes denote the same concept "
    "(translation, abbreviation, spelling or inflection variant, transliteration). "
    'Answer JSON: {"pairs": [{"id": <int>, "same": true|false}]} covering every id.'
)


@dataclass
class Judge:
    client: LlmClient
    batch_nodes: int = 40
    batch_pairs: int = 30
    unit_text_bytes: int = 8500

    def label_nodes(self, graph: nx.Graph) -> dict[str, str]:
        nodes = sorted(graph.nodes, key=str)
        labels: dict[str, str] = {}
        for start in range(0, len(nodes), self.batch_nodes):
            batch = nodes[start : start + self.batch_nodes]
            lines = [
                f"{i}. {name} | {graph.nodes[name]['type']} | {graph.nodes[name]['description'][:110]}"
                for i, name in enumerate(batch)
            ]
            answer = self.client.complete_json(NODE_SYSTEM, "\n".join(lines))
            by_id = {int(item["id"]): str(item["label"]) for item in _items(answer, "labels") if "label" in item}
            for i, name in enumerate(batch):
                labels[name] = by_id.get(i, "unlabeled")
        return labels

    def judge_units(self, bundle: GraphBundle) -> list[dict[str, object]]:
        entities = bundle.artifacts.entities
        relationships = bundle.artifacts.relationships
        titles = dict(zip(bundle.artifacts.documents["id"], bundle.artifacts.documents["title"], strict=True))
        results: list[dict[str, object]] = []
        for unit in bundle.artifacts.text_units.itertuples():
            unit_entities = entities[entities["text_unit_ids"].map(lambda ids, u=unit.id: u in list(ids))]
            unit_relations = relationships[relationships["text_unit_ids"].map(lambda ids, u=unit.id: u in list(ids))]
            entity_lines = [f"- {row.title} [{row.type}]" for row in unit_entities.itertuples()]
            relation_lines = [
                f"- {row.source} -> {row.target}: {str(row.description)[:90]}" for row in unit_relations.itertuples()
            ]
            text = _truncate_bytes(str(unit.text), self.unit_text_bytes)
            user = (
                f"FRAGMENT:\n{text}\n\nENTITIES ({len(entity_lines)}):\n"
                + "\n".join(entity_lines)
                + f"\n\nRELATIONS ({len(relation_lines)}):\n"
                + "\n".join(relation_lines)
            )
            user = _truncate_bytes(user, 15500)
            answer = self.client.complete_json(UNIT_SYSTEM, user)
            book = titles[unit.document_id].split("_")[0].removesuffix(".md")
            results.append({"unit": unit.id, "book": book, "entities": len(entity_lines), **answer})
        return results

    def duplicates(self, graph: nx.Graph, vectors: dict[str, np.ndarray], threshold: float) -> dict[str, object]:
        names = [name for name in graph.nodes if name in vectors]
        matrix = np.stack([vectors[name] for name in names])
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        similarity = matrix @ matrix.T
        candidates = [
            (names[i], names[j], float(similarity[i, j]))
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if similarity[i, j] >= threshold
        ]
        confirmed: list[tuple[str, str]] = []
        for start in range(0, len(candidates), self.batch_pairs):
            batch = candidates[start : start + self.batch_pairs]
            lines = [f"{i}. {a} || {b}" for i, (a, b, _) in enumerate(batch)]
            answer = self.client.complete_json(DUPLICATE_SYSTEM, "\n".join(lines))
            same = {int(item["id"]) for item in _items(answer, "pairs") if item.get("same") is True}
            confirmed.extend((a, b) for i, (a, b, _) in enumerate(batch) if i in same)
        cross = [(a, b) for a, b in confirmed if {graph.nodes[a]["lang"], graph.nodes[b]["lang"]} == {"en", "ru"}]
        return {
            "candidates": len(candidates),
            "confirmed": len(confirmed),
            "confirmed_cross_language": len(cross),
            "duplicate_node_share": round(len({n for pair in confirmed for n in pair}) / max(len(names), 1), 4),
            "examples": [f"{a} = {b}" for a, b in confirmed[:15]],
        }


def summarize_units(results: list[dict[str, object]]) -> dict[str, object]:
    def mean(key: str) -> float | None:
        values = [float(r[key]) for r in results if isinstance(r.get(key), int | float)]
        return round(statistics.fmean(values), 3) if values else None

    def total(key: str) -> int:
        return int(sum(r.get(key) or 0 for r in results if isinstance(r.get(key), int | float)))

    entities = sum(int(r["entities"]) for r in results)
    return {
        "units": len(results),
        "completeness_mean": mean("completeness"),
        "domain_coverage_mean": mean("domain_coverage"),
        "formula_table_integrity_mean": mean("formula_table_integrity"),
        "contradictions_total": total("contradictions"),
        "coreference_duplicates_total": total("coreference_duplicates"),
        "noise_entities_total": total("noise_entities"),
        "noise_entities_share": round(total("noise_entities") / max(entities, 1), 4),
        "entity_mentions": entities,
    }


def label_summary(labels: dict[str, str]) -> dict[str, object]:
    counts = {label: sum(1 for value in labels.values() if value == label) for label in ("core", "peripheral", "noise")}
    total = max(len(labels), 1)
    return {**counts, "noise_share": round(counts["noise"] / total, 4), "core_share": round(counts["core"] / total, 4)}


def _truncate_bytes(text: str, limit: int) -> str:
    encoded = text.encode()
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode(errors="ignore") + "\n[...]"


def dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _items(answer: dict, key: str) -> list[dict]:
    items = answer.get(key, [])
    if isinstance(items, dict):
        items = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "value": v} for k, v in items.items()]
    return [item for item in items if isinstance(item, dict) and str(item.get("id", "")).isdigit()]


CONSISTENCY_SYSTEM = (
    "You are an expert in physical metallurgy and steel rolling. You check one vertex of a knowledge graph built from "
    "texts on controlled rolling of steel. You get the vertex, its neighbourhood collected by a breadth-first "
    "traversal (neighbour vertices with descriptions and the relations between them) and the source text fragment "
    "the vertex was extracted from. Judge strictly and only against the source fragment. Answer JSON with keys: "
    '"definition" ("full" if the vertex and its neighbourhood define the term completely and correctly, "partial" if '
    'important parts are missing, "none" if it is not defined or not a real term), '
    '"contradictions" (int: statements in the vertex or its relations that contradict or distort the source), '
    '"comment" (one short sentence).'
)


def consistency_prompt(graph: nx.Graph, node: str, nodes: list[str], source: str, limit: int = 3500) -> str:
    lines = [f"VERTEX: {node} [{graph.nodes[node].get('type', '')}]: {graph.nodes[node].get('description', '')[:500]}"]
    lines.append("NEIGHBOURHOOD:")
    lines.extend(
        f"- {n} [{graph.nodes[n].get('type', '')}]: {graph.nodes[n].get('description', '')[:250]}"
        for n in nodes
        if n != node
    )
    lines.append("RELATIONS:")
    members = set(nodes)
    lines.extend(
        f"- {a} — {b}: {str(data.get('description', ''))[:200]}"
        for a, b, data in graph.edges(nodes, data=True)
        if a in members and b in members
    )
    lines.append(f"SOURCE FRAGMENT:\n{_truncate_bytes(source, limit)}")
    return _truncate_bytes("\n".join(lines), 15000)


def check_consistency(
    client: LlmClient, graph: nx.Graph, node: str, nodes: list[str], source: str
) -> dict[str, object]:
    answer = client.complete_json(CONSISTENCY_SYSTEM, consistency_prompt(graph, node, nodes, source))
    definition = str(answer.get("definition", "none")).lower()
    contradictions = answer.get("contradictions", 0)
    return {
        "node": node,
        "definition": definition if definition in {"full", "partial", "none"} else "none",
        "contradictions": int(contradictions) if isinstance(contradictions, int | float) else 0,
        "comment": str(answer.get("comment", ""))[:300],
    }
