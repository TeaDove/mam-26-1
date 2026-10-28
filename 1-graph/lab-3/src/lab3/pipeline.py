import json
import logging
import re
import statistics
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import TypeAdapter
from sentence_transformers import SentenceTransformer

from lab3.answer import context_text, generate, graph_text, judge
from lab3.evaluate import first_relevant_rank, retrieval_metrics, run_metrics
from lab3.gnn import GnnConfig, LinkEncoder, embed, train_link_prediction
from lab3.graph import KnowledgeGraph, load_graph, relevant_vertices, vertex_text
from lab3.llm import LlmClient
from lab3.models import (
    LinkMetrics,
    QueryGraph,
    Question,
    QuestionResult,
    QuestionSet,
    RagReport,
    RetrievalMetrics,
    RunMetrics,
)
from lab3.query import extract_query_graph
from lab3.retrieve import Retrieval, RetrievalConfig, combine, query_edges, retrieve, source_units
from lab3.settings import Settings

log = logging.getLogger(__name__)
QUERY_GRAPHS = TypeAdapter(dict[str, QueryGraph])
LINK_METRICS = TypeAdapter(list[LinkMetrics])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def questions(settings: Settings) -> list[Question]:
    return QuestionSet.model_validate(yaml.safe_load(settings.questions_path.read_text(encoding="utf-8"))).questions


def encoder_model(settings: Settings) -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model, device=settings.device)


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def gnn_config(settings: Settings, kind: str, seed: int) -> GnnConfig:
    return GnnConfig(
        kind=kind,
        hidden=settings.hidden,
        heads=settings.heads,
        dropout=settings.dropout,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        epochs=settings.epochs,
        patience=settings.patience,
        seed=seed,
    )


def retrieval_config(settings: Settings) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=settings.top_k,
        seed_vertices=settings.seed_vertices,
        context_vertices=settings.context_vertices,
        relation_bonus=settings.relation_bonus,
        two_hop_bonus=settings.two_hop_bonus,
    )


def run_features(settings: Settings) -> None:
    model = encoder_model(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for graph in settings.graphs:
        kg = load_graph(graph, settings.graphrag_output(graph))
        np.save(settings.data_dir / f"{graph}_features.npy", encode(model, kg.texts(settings.description_chars)))
        log.info("%s: %d vertices, %d edges encoded", graph, len(kg.nodes), kg.graph.number_of_edges())


def model_path(settings: Settings, graph: str, kind: str, seed: int) -> Path:
    return settings.data_dir / "models" / f"{graph}_{kind}_{seed}.pt"


def run_train(settings: Settings) -> None:
    metrics: list[LinkMetrics] = []
    (settings.data_dir / "models").mkdir(parents=True, exist_ok=True)
    for graph in settings.graphs:
        kg = load_graph(graph, settings.graphrag_output(graph))
        x = np.load(settings.data_dir / f"{graph}_features.npy")
        edges = kg.edge_array()
        for kind in settings.encoders:
            for seed in settings.gnn_seeds:
                model, result = train_link_prediction(x, edges, gnn_config(settings, kind, seed), settings.device)
                torch.save(model.state_dict(), model_path(settings, graph, kind, seed))
                metrics.append(
                    LinkMetrics(
                        graph=graph,
                        encoder=kind,
                        seed=seed,
                        vertices=len(kg.nodes),
                        edges=edges.shape[1],
                        best_epoch=result.best_epoch,
                        cosine_test_auc=result.cosine_test_auc,
                        val_auc=result.val_auc,
                        test_auc=result.test_auc,
                        test_ap=result.test_ap,
                    )
                )
                log.info("%s %s seed %d: %s", graph, kind, seed, result)
    write_json(settings.metrics_dir / "link_prediction.json", [m.model_dump() for m in metrics])


def llm(settings: Settings, model: str) -> LlmClient:
    return LlmClient(
        base_url=settings.llm_base_url,
        model=model,
        cache_dir=settings.data_dir / "llm_cache",
        api_key=settings.llm_api_key,
        api_key_header=settings.llm_api_key_header,
        api_key_prefix=settings.llm_api_key_prefix,
        timeout=settings.llm_timeout,
    )


def run_queries(settings: Settings) -> None:
    client = llm(settings, settings.llm_model)
    graphs = {q.id: extract_query_graph(client, q.question) for q in questions(settings)}
    write_json(settings.root / "results" / "query_graphs.json", {k: v.model_dump() for k, v in graphs.items()})
    log.info("query graphs written for %d questions", len(graphs))


def load_encoders(settings: Settings, graph: str, kind: str, in_dim: int) -> list[LinkEncoder]:
    models = []
    for seed in settings.gnn_seeds:
        model = LinkEncoder(kind, in_dim, settings.hidden, settings.heads, settings.dropout).to(settings.device)
        model.load_state_dict(torch.load(model_path(settings, graph, kind, seed), map_location=settings.device))
        models.append(model)
    return models


def blocks(
    settings: Settings, retriever: str, x: np.ndarray, edges: np.ndarray, encoders: list[LinkEncoder]
) -> np.ndarray:
    if retriever == "bge-m3":
        return x
    share = settings.hybrid_weight if retriever == "hybrid" else 1.0
    parts = [(share / len(encoders), embed(model, x, edges, settings.device)) for model in encoders]
    return combine([(1 - share, x), *parts] if retriever == "hybrid" else parts)


def retriever_encoders(settings: Settings, graph: str, retriever: str) -> list[LinkEncoder]:
    if retriever == "bge-m3":
        return []
    return load_encoders(settings, graph, "gat" if retriever == "hybrid" else retriever, 1024)


def query_vectors(
    settings: Settings, sbert: SentenceTransformer, query: QueryGraph, retriever: str, encoders: list[LinkEncoder]
) -> np.ndarray:
    texts = [vertex_text(e.name, e.description, settings.description_chars) for e in query.entities]
    return blocks(settings, retriever, encode(sbert, texts), query_edges(query), encoders)


def answer_question(
    settings: Settings, kg: KnowledgeGraph, question: Question, query: QueryGraph, found: Retrieval, retriever: str
) -> QuestionResult:
    units = source_units(kg, found.seeds, found.context, settings.source_units)
    context = context_text(kg, found.context, units, settings.source_chars)
    answer = generate(llm(settings, settings.llm_model), question.question, context)
    verdict = judge(llm(settings, settings.judge_model), question.question, question.answer, answer)
    relevant = relevant_vertices(kg, question.evidence)
    key = re.compile(question.key, re.IGNORECASE)
    return QuestionResult(
        id=question.id,
        graph=kg.name,
        retriever=retriever,
        question=question.question,
        reference=question.answer,
        answer=answer,
        verdict=verdict.verdict,
        comment=verdict.comment,
        query_entities=[e.name for e in query.entities],
        seeds=found.seeds,
        context=found.context,
        source_units=units,
        relevant_vertices=len(relevant),
        first_relevant_rank=first_relevant_rank(found.ranking, relevant),
        context_has_relevant=bool(relevant & set(found.context)),
        key_in_graph=bool(key.search(graph_text(kg, kg.nodes))),
        key_in_context=bool(key.search(context)),
    )


def run_retrieval(settings: Settings) -> None:
    sbert = encoder_model(settings)
    queries = QUERY_GRAPHS.validate_json((settings.root / "results" / "query_graphs.json").read_text(encoding="utf-8"))
    runs: list[RetrievalMetrics] = []
    for graph in settings.graphs:
        kg = load_graph(graph, settings.graphrag_output(graph))
        x = np.load(settings.data_dir / f"{graph}_features.npy")
        for retriever in settings.retrievers:
            encoders = retriever_encoders(settings, graph, retriever)
            vectors = blocks(settings, retriever, x, kg.edge_array(), encoders)
            ranks: list[int | None] = []
            context_hits: list[bool] = []
            key_hits: list[bool] = []
            for question in questions(settings):
                query = queries[question.id]
                query_x = query_vectors(settings, sbert, query, retriever, encoders)
                found = retrieve(kg, vectors, query, query_x, retrieval_config(settings))
                relevant = relevant_vertices(kg, question.evidence)
                ranks.append(first_relevant_rank(found.ranking, relevant) if relevant else None)
                context_hits.append(bool(relevant & set(found.context)))
                units = source_units(kg, found.seeds, found.context, settings.source_units)
                context = context_text(kg, found.context, units, settings.source_chars)
                key_hits.append(bool(re.compile(question.key, re.IGNORECASE).search(context)))
            runs.append(retrieval_metrics(graph, retriever, ranks, context_hits, key_hits))
            log.info("%s", runs[-1])
    write_json(settings.metrics_dir / "retrieval.json", [r.model_dump() for r in runs])


def run_rag(settings: Settings) -> None:
    sbert = encoder_model(settings)
    queries = QUERY_GRAPHS.validate_json((settings.root / "results" / "query_graphs.json").read_text(encoding="utf-8"))
    runs: list[RunMetrics] = []
    for graph in settings.graphs:
        kg = load_graph(graph, settings.graphrag_output(graph))
        x = np.load(settings.data_dir / f"{graph}_features.npy")
        for retriever in settings.retrievers:
            encoders = retriever_encoders(settings, graph, retriever)
            vectors = blocks(settings, retriever, x, kg.edge_array(), encoders)
            results = []
            for question in questions(settings):
                query = queries[question.id]
                query_x = query_vectors(settings, sbert, query, retriever, encoders)
                found = retrieve(kg, vectors, query, query_x, retrieval_config(settings))
                results.append(answer_question(settings, kg, question, query, found, retriever))
            path = settings.root / "results" / "answers" / f"{graph}_{retriever}.json"
            write_json(path, [r.model_dump() for r in results])
            runs.append(run_metrics(graph, retriever, results))
            log.info("%s", runs[-1])
    link = LINK_METRICS.validate_json((settings.metrics_dir / "link_prediction.json").read_text(encoding="utf-8"))
    report = RagReport(runs=runs, link_prediction=link)
    write_json(settings.metrics_dir / "rag.json", report.model_dump())
    (settings.metrics_dir / "rag.md").write_text(report_markdown(report), encoding="utf-8")


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f} ± {statistics.pstdev(values):.3f}"


def link_rows(metrics: list[LinkMetrics]) -> str:
    groups: dict[tuple[str, str], list[LinkMetrics]] = {}
    for m in metrics:
        groups.setdefault((m.graph, m.encoder), []).append(m)
    return "".join(
        f"| {graph} | {encoder} | {len(items)} | {mean_std([m.cosine_test_auc for m in items])} | "
        f"{mean_std([m.val_auc for m in items])} | {mean_std([m.test_auc for m in items])} | "
        f"{mean_std([m.test_ap for m in items])} |\n"
        for (graph, encoder), items in groups.items()
    )


def report_markdown(report: RagReport) -> str:
    header = (
        "| graph | retriever | MRR | Hit@1 | Hit@3 | Hit@10 | context recall | key in graph | key in context | "
        "correct | correct+partial |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = "".join(
        f"| {r.graph} | {r.retriever} | {r.mrr:.3f} | {r.hit_at_1:.2f} | {r.hit_at_3:.2f} | {r.hit_at_10:.2f} | "
        f"{r.context_recall:.2f} | {r.key_in_graph_share:.2f} | {r.key_in_context_share:.2f} | "
        f"{r.correct_share:.2f} | {r.correct_or_partial_share:.2f} |\n"
        for r in report.runs
    )
    return (
        f"## RAG\n\n{header}{rows}\n## Link prediction (mean ± std over seeds)\n\n"
        "| graph | encoder | seeds | bge-m3 cosine test ROC-AUC | val ROC-AUC | test ROC-AUC | test AP |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        f"{link_rows(report.link_prediction)}"
    )


def run_ask(settings: Settings, question: str, graph: str, retriever: str) -> str:
    sbert = encoder_model(settings)
    kg = load_graph(graph, settings.graphrag_output(graph))
    query = extract_query_graph(llm(settings, settings.llm_model), question)
    encoders = retriever_encoders(settings, graph, retriever)
    x = np.load(settings.data_dir / f"{graph}_features.npy")
    vectors = blocks(settings, retriever, x, kg.edge_array(), encoders)
    query_x = query_vectors(settings, sbert, query, retriever, encoders)
    found = retrieve(kg, vectors, query, query_x, retrieval_config(settings))
    units = source_units(kg, found.seeds, found.context, settings.source_units)
    context = context_text(kg, found.context, units, settings.source_chars)
    answer = generate(llm(settings, settings.llm_model), question, context)
    return f"QUERY GRAPH: {query.model_dump_json()}\nSEEDS: {found.seeds}\n{context}\n\nANSWER: {answer}"
