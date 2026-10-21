import re

from lab3.graph import KnowledgeGraph
from lab3.llm import LlmClient
from lab3.models import Answer, Verdict

NO_ANSWER = "В контексте нет ответа."

ANSWER_SYSTEM = (
    "You answer questions about controlled rolling of steel using ONLY the knowledge-graph context given by the "
    "user: entities with descriptions, relations between them and, if present, the source text the entities were "
    "extracted from. Do not use outside knowledge. If the context "
    f'does not contain the answer, reply exactly "{NO_ANSWER}". Answer in the language of the question, in 1–3 '
    'sentences, keep numbers and units as in the context. Answer JSON: {"answer": "..."}.'
)

JUDGE_SYSTEM = (
    "You grade an answer to a question about controlled rolling of steel against the reference answer. "
    '"correct": the answer states the key fact of the reference (numbers and names must match; wording and '
    'language may differ; extra correct details are fine). "partial": only part of the key fact, or the key fact '
    'mixed with a contradicting claim. "wrong": the key fact is missing or wrong, or the answer says there is no '
    'answer. Answer JSON: {"verdict": "correct" | "partial" | "wrong", "comment": "one short sentence"}.'
)

VERDICTS = ("correct", "partial", "wrong")


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def context_text(kg: KnowledgeGraph, context: list[str], units: list[str] | None = None, unit_chars: int = 5000) -> str:
    graph = graph_text(kg, context)
    sources = "\n\n".join(clip(kg.unit_text[unit], unit_chars) for unit in units or [])
    return f"{graph}\nSOURCE TEXT:\n{sources}" if sources else graph


def graph_text(kg: KnowledgeGraph, context: list[str]) -> str:
    position = {node: i for i, node in enumerate(context)}
    entities = (
        f"- {node} [{kg.graph.nodes[node].get('type', '')}]: {clip(kg.graph.nodes[node].get('description', ''), 400)}"
        for node in context
    )
    relations = (
        f"- {a} — {b}: {clip(data.get('description', ''), 250)}"
        for a in context
        for b, data in kg.graph[a].items()
        if position.get(b, -1) > position[a]
    )
    return f"ENTITIES:\n{'\n'.join(entities)}\nRELATIONS:\n{'\n'.join(relations)}"


def generate(client: LlmClient, question: str, context: str) -> str:
    return client.complete(ANSWER_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION: {question}", Answer).answer.strip()


def judge(client: LlmClient, question: str, reference: str, answer: str) -> Verdict:
    verdict = client.complete(JUDGE_SYSTEM, f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}", Verdict)
    label = verdict.verdict.strip().lower()
    return Verdict(verdict=label if label in VERDICTS else "wrong", comment=clip(verdict.comment, 300))
