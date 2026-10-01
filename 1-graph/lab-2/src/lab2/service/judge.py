import networkx as nx

from lab2.dto.models import ConsistencyVerdict, JudgeAnswer
from lab2.supplier.llm import LlmClient


def _truncate_bytes(text: str, limit: int) -> str:
    encoded = text.encode()
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode(errors="ignore") + "\n[...]"


CONSISTENCY_SYSTEM = (
    "You are an expert in physical metallurgy and steel rolling. You check one vertex of a knowledge graph built from "
    "texts on controlled rolling of steel. You get the vertex, its neighbourhood collected by a breadth-first "
    "traversal (neighbour vertices with descriptions and the relations between them) and the source text fragment "
    "the vertex was extracted from. Judge strictly and only against the source fragment. Answer JSON with keys: "
    '"definition" ("full" if the vertex and its neighbourhood define the term completely and correctly, "partial" if '
    'important parts are missing, "none" if it is not defined or not a real term), '
    '"contradictions" (int: statements in the vertex or its relations that conflict with the source or distort it; '
    "a statement the source does not mention is incompleteness, not a contradiction), "
    '"comment" (one short sentence).'
)


def consistency_prompt(graph: nx.Graph, node: str, nodes: list[str], source: str, limit: int = 15000) -> str:
    fragment = f"SOURCE FRAGMENT:\n{_truncate_bytes(source, 7000)}"
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
    context = _truncate_bytes("\n".join(lines), limit - len(fragment.encode()) - 1)
    return f"{context}\n{fragment}"


def check_consistency(
    client: LlmClient, graph: nx.Graph, node: str, nodes: list[str], source: str
) -> ConsistencyVerdict:
    answer = client.complete(CONSISTENCY_SYSTEM, consistency_prompt(graph, node, nodes, source), JudgeAnswer)
    definition = answer.definition.lower()
    return ConsistencyVerdict(
        node=node,
        definition=definition if definition in {"full", "partial", "none"} else "none",
        contradictions=answer.contradictions,
        comment=answer.comment[:300],
    )
