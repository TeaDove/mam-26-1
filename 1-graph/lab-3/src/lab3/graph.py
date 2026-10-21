import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class KnowledgeGraph:
    name: str
    graph: nx.Graph
    nodes: list[str]
    units: dict[str, list[str]]
    unit_text: dict[str, str]

    @property
    def index(self) -> dict[str, int]:
        return {node: i for i, node in enumerate(self.nodes)}

    def edge_array(self) -> np.ndarray:
        index = self.index
        pairs = [(index[a], index[b]) for a, b in self.graph.edges if a != b]
        if not pairs:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(pairs, dtype=np.int64).T

    def texts(self, limit: int) -> list[str]:
        return [vertex_text(node, self.graph.nodes[node].get("description", ""), limit) for node in self.nodes]


def vertex_text(name: str, description: str, limit: int) -> str:
    description = re.sub(r"\s+", " ", description).strip()[:limit]
    return f"{name}: {description}" if description else name


def build_graph(name: str, entities: pd.DataFrame, relationships: pd.DataFrame, units: pd.DataFrame) -> KnowledgeGraph:
    graph = nx.Graph()
    vertex_units: dict[str, list[str]] = {}
    for row in entities.itertuples():
        graph.add_node(row.title, type=str(row.type or "").lower(), description=str(row.description or ""))
        vertex_units[row.title] = [str(unit) for unit in row.text_unit_ids]
    for row in relationships.itertuples():
        for endpoint in (row.source, row.target):
            if endpoint not in graph:
                graph.add_node(endpoint, type="", description="")
                vertex_units[endpoint] = []
        description = str(row.description or "")
        if graph.has_edge(row.source, row.target):
            edge = graph[row.source][row.target]
            edge["description"] = f"{edge['description']} {description}".strip()
            edge["weight"] += float(row.weight)
        else:
            graph.add_edge(row.source, row.target, description=description, weight=float(row.weight))
    unit_text = {str(row.id): str(row.text) for row in units.itertuples()}
    return KnowledgeGraph(name=name, graph=graph, nodes=sorted(graph.nodes), units=vertex_units, unit_text=unit_text)


def load_graph(name: str, output: Path) -> KnowledgeGraph:
    return build_graph(
        name,
        pd.read_parquet(output / "entities.parquet"),
        pd.read_parquet(output / "relationships.parquet"),
        pd.read_parquet(output / "text_units.parquet"),
    )


def relevant_vertices(kg: KnowledgeGraph, evidence: list[str]) -> set[str]:
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in evidence]
    matching = {
        unit
        for unit, text in kg.unit_text.items()
        if all(pattern.search(re.sub(r"\s+", " ", text)) for pattern in patterns)
    }
    return {node for node in kg.nodes if matching.intersection(kg.units.get(node, []))}
