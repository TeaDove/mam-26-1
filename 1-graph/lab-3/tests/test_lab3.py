import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.utils import to_undirected

from lab3.answer import NO_ANSWER, context_text, judge
from lab3.evaluate import first_relevant_rank, retrieval_metrics, run_metrics
from lab3.gnn import GnnConfig, decode, embed, known_pairs, sample_negatives, train_link_prediction
from lab3.graph import KnowledgeGraph, build_graph, relevant_vertices
from lab3.llm import parse_json
from lab3.models import Answer, QueryEntity, QueryGraph, QueryRelation, QuestionResult, Verdict
from lab3.query import extract_query_graph
from lab3.retrieve import RetrievalConfig, combine, expand, query_edges, retrieve, source_units

CONFIG = RetrievalConfig(top_k=2, seed_vertices=1, context_vertices=3, relation_bonus=0.1, two_hop_bonus=0.05)


def toy_graph() -> KnowledgeGraph:
    entities = pd.DataFrame(
        {
            "title": ["NB", "RECRYSTALLIZATION", "FERRITE", "USINOR"],
            "type": ["alloying element", "phenomenon", "phase", "organization"],
            "description": ["Niobium retards recrystallization", "Recrystallization of austenite", "Ferrite", "Firm"],
            "text_unit_ids": [["u1"], ["u1"], ["u2"], ["u3"]],
        }
    )
    relationships = pd.DataFrame(
        {
            "source": ["NB", "RECRYSTALLIZATION", "NB"],
            "target": ["RECRYSTALLIZATION", "FERRITE", "RECRYSTALLIZATION"],
            "description": ["Nb retards it", "leads to ferrite", "strongly"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    units = pd.DataFrame({"id": ["u1", "u2", "u3"], "text": ["Nb has a  retarding effect", "ferrite", "USINOR"]})
    return build_graph("toy", entities, relationships, units)


class FakeClient:
    def __init__(self, reply: object) -> None:
        self.reply = reply

    def complete(self, system: str, user: str, model: type) -> object:
        return self.reply


def test_build_graph_merges_duplicate_edges() -> None:
    kg = toy_graph()
    assert kg.graph.number_of_edges() == 2
    assert kg.graph["NB"]["RECRYSTALLIZATION"]["weight"] == 4.0
    assert kg.edge_array().shape == (2, 2)


def test_relevant_vertices_require_all_evidence_patterns() -> None:
    kg = toy_graph()
    assert relevant_vertices(kg, ["Nb", "retard"]) == {"NB", "RECRYSTALLIZATION"}
    assert relevant_vertices(kg, ["Nb", "absent"]) == set()


def test_query_edges_and_extraction_fallback() -> None:
    query = QueryGraph(
        entities=[QueryEntity(name="A"), QueryEntity(name="B")],
        relations=[QueryRelation(source="A", target="B"), QueryRelation(source="A", target="C")],
    )
    assert query_edges(query).tolist() == [[0], [1]]
    empty = extract_query_graph(FakeClient(QueryGraph()), "What is it?")
    assert [e.name for e in empty.entities] == ["What is it?"]
    cleaned = extract_query_graph(FakeClient(query), "q")
    assert len(cleaned.relations) == 1


def test_retrieve_ranks_by_similarity_and_relation_bonus() -> None:
    kg = toy_graph()
    vectors = np.eye(4, dtype=np.float32)
    query = QueryGraph(
        entities=[QueryEntity(name="NIOBIUM"), QueryEntity(name="RECRYST")],
        relations=[QueryRelation(source="NIOBIUM", target="RECRYST")],
    )
    query_vectors = np.array([vectors[kg.index["NB"]], 0.9 * vectors[kg.index["RECRYSTALLIZATION"]]])
    found = retrieve(kg, vectors, query, query_vectors, CONFIG)
    assert found.ranking[:2] == ["NB", "RECRYSTALLIZATION"]
    assert found.seeds == ["NB"]
    assert found.context == ["NB", "RECRYSTALLIZATION", "FERRITE"]


def test_expand_respects_limit() -> None:
    kg = toy_graph()
    score = dict.fromkeys(kg.nodes, 0.0)
    assert expand(kg, ["NB"], score, 2) == ["NB", "RECRYSTALLIZATION"]


def test_source_units_prefer_units_shared_by_context() -> None:
    kg = toy_graph()
    assert source_units(kg, ["FERRITE", "NB"], ["NB", "RECRYSTALLIZATION", "FERRITE"], 1) == ["u1"]
    assert source_units(kg, ["FERRITE", "NB"], [], 5) == ["u2", "u1"]


def test_context_text_appends_source_text() -> None:
    text = context_text(toy_graph(), ["NB"], ["u1"], 10)
    assert text.endswith("SOURCE TEXT:\nNb has a r…")


def test_relations_follow_context_order() -> None:
    kg = toy_graph()
    text = context_text(kg, ["FERRITE", "RECRYSTALLIZATION", "NB"])
    assert text.split("RELATIONS:\n")[1].splitlines() == [
        "- FERRITE — RECRYSTALLIZATION: leads to ferrite",
        "- RECRYSTALLIZATION — NB: Nb retards it strongly",
    ]


def test_combine_weights_cosine_blocks() -> None:
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    left = combine([(0.7, a), (0.3, a)])
    right = combine([(0.7, a), (0.3, b)])
    assert (left @ left.T).item() == pytest.approx(1.0)
    assert (left @ right.T).item() == pytest.approx(0.7)


def test_context_text_lists_entities_and_relations() -> None:
    text = context_text(toy_graph(), ["NB", "RECRYSTALLIZATION"])
    assert "- NB [alloying element]: Niobium retards recrystallization" in text
    assert "RELATIONS:\n- NB — RECRYSTALLIZATION: Nb retards it strongly" in text


def test_parse_json_handles_fences_and_preamble() -> None:
    assert parse_json(Answer, '```json\n{"answer": "ok"}\n```').answer == "ok"
    assert parse_json(Answer, 'Result: {"answer": "ok"} end').answer == "ok"


def test_judge_normalizes_unknown_verdicts() -> None:
    assert judge(FakeClient(Verdict(verdict=" Correct ")), "q", "r", "a").verdict == "correct"
    assert judge(FakeClient(Verdict(verdict="maybe")), "q", "r", NO_ANSWER).verdict == "wrong"


def result(rank: int | None, relevant: int, verdict: str) -> QuestionResult:
    return QuestionResult(
        id="x",
        graph="g",
        retriever="r",
        question="q",
        reference="ref",
        answer="a",
        verdict=verdict,
        comment="",
        query_entities=[],
        seeds=[],
        context=[],
        source_units=[],
        relevant_vertices=relevant,
        first_relevant_rank=rank,
        context_has_relevant=bool(rank and rank <= 3),
        key_in_graph=relevant > 0,
        key_in_context=verdict == "correct",
    )


def test_retrieval_metrics_counts() -> None:
    metrics = retrieval_metrics("g", "r", [1, 3, None], [True, False, True], [True, False, True])
    assert metrics.with_relevant == 2
    assert metrics.mrr == pytest.approx(0.6667)
    assert metrics.hit_at_1 == 0.5
    assert metrics.hit_at_10 == 1.0
    assert metrics.context_recall == 0.5
    assert metrics.key_in_context_share == pytest.approx(0.6667)


def test_cosine_decode_is_bounded() -> None:
    z = torch.randn(8, 4)
    logits = decode(z, torch.tensor([[0, 1], [2, 3]]))
    assert torch.all(logits.abs() <= 10.0 + 1e-5)


def test_negative_sampling_excludes_known_edges() -> None:
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    known = known_pairs(to_undirected(edges), 6, "cpu")
    positive = torch.tensor([[0, 2], [1, 3]])
    for _ in range(20):
        negatives = sample_negatives(positive, 6, known)
        assert not known[negatives[0], negatives[1]].any()
        assert not (negatives[0] == negatives[1]).any()


def test_run_metrics() -> None:
    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3
    metrics = run_metrics("g", "r", [result(1, 2, "correct"), result(4, 1, "partial"), result(None, 0, "wrong")])
    assert metrics.with_relevant == 2
    assert metrics.mrr == pytest.approx(0.625)
    assert metrics.hit_at_1 == 0.5
    assert metrics.hit_at_10 == 1.0
    assert metrics.context_recall == 0.5
    assert metrics.key_in_graph_share == pytest.approx(0.6667)
    assert metrics.key_in_context_share == pytest.approx(0.3333)
    assert metrics.correct_share == pytest.approx(0.3333)
    assert metrics.correct_or_partial_share == pytest.approx(0.6667)


@pytest.mark.parametrize("kind", ["gat", "sage"])
def test_link_prediction_trains_and_embeds(kind: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 16)).astype(np.float32)
    edges = np.array([[i, (i + 1) % 60] for i in range(60)] + [[i, (i + 7) % 60] for i in range(60)]).T
    config = GnnConfig(kind, 8, 2, 0.1, 0.01, 5e-4, 20, 5, 0)
    model, result = train_link_prediction(x, edges, config, "cpu")
    assert 0.0 <= result.test_auc <= 1.0
    vectors = embed(model, x, edges, "cpu")
    assert vectors.shape == (60, 8)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    single = embed(model, x[:1], np.zeros((2, 0), dtype=np.int64), "cpu")
    assert single.shape == (1, 8)
