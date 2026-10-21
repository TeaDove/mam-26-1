from lab3.models import QuestionResult, RunMetrics


def first_relevant_rank(ranking: list[str], relevant: set[str]) -> int | None:
    return next((rank for rank, node in enumerate(ranking, start=1) if node in relevant), None)


def share(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def run_metrics(graph: str, retriever: str, results: list[QuestionResult]) -> RunMetrics:
    scored = [r for r in results if r.relevant_vertices]
    ranks = [r.first_relevant_rank for r in scored]
    return RunMetrics(
        graph=graph,
        retriever=retriever,
        questions=len(results),
        with_relevant=len(scored),
        mrr=round(sum(1 / rank for rank in ranks if rank) / len(scored), 4) if scored else 0.0,
        hit_at_1=share(sum(bool(rank and rank <= 1) for rank in ranks), len(scored)),
        hit_at_3=share(sum(bool(rank and rank <= 3) for rank in ranks), len(scored)),
        hit_at_10=share(sum(bool(rank and rank <= 10) for rank in ranks), len(scored)),
        context_recall=share(sum(r.context_has_relevant for r in scored), len(scored)),
        key_in_graph_share=share(sum(r.key_in_graph for r in results), len(results)),
        key_in_context_share=share(sum(r.key_in_context for r in results), len(results)),
        correct_share=share(sum(r.verdict == "correct" for r in results), len(results)),
        correct_or_partial_share=share(sum(r.verdict in ("correct", "partial") for r in results), len(results)),
    )
