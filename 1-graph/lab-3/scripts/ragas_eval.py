import json
import re
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import typer
import yaml
from loguru import logger
from ragas import SingleTurnSample
from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall

GRAPHS = ("dirty", "clean")
RETRIEVERS = ("bge-m3", "gat", "sage", "hybrid")
APP = typer.Typer()
LAB_OPTION = typer.Option(..., exists=True, file_okay=False)
OUTPUT_OPTION = typer.Option(...)


def reference_units(index: Path, questions: list[dict[str, Any]]) -> tuple[set[str], dict[str, list[str]]]:
    units = pd.read_parquet(index / "text_units.parquet", columns=["id", "document_id", "text"])
    docs = pd.read_parquet(index / "documents.parquet", columns=["id", "title"])
    titles = dict(zip(docs["id"], docs["title"], strict=True))
    chunks = [
        (str(row.id), Path(titles[row.document_id]).stem.split("_", 1)[0], re.sub(r"\s+", " ", row.text))
        for row in units.itertuples(index=False)
    ]
    refs = {}
    for question in questions:
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in question["evidence"]]
        refs[question["id"]] = [
            unit for unit, book, text in chunks if book == question["book"] and all(p.search(text) for p in patterns)
        ]
    return {unit for unit, _, _ in chunks}, refs


def score_run(
    path: Path, graph: str, retriever: str, units: set[str], refs: dict[str, list[str]]
) -> list[dict[str, Any]]:
    precision = IDBasedContextPrecision()
    recall = IDBasedContextRecall()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if len(saved) != len(refs) or {row["id"] for row in saved} != set(refs):
        raise ValueError(f"Unexpected question IDs in {path}")
    rows = []
    for row in saved:
        retrieved, reference = row["source_units"], refs[row["id"]]
        if row["graph"] != graph or row["retriever"] != retriever or set(retrieved) - units:
            raise ValueError(f"Unexpected retrieval record in {path} for {row['id']}")
        sample = SingleTurnSample(retrieved_context_ids=retrieved, reference_context_ids=reference)
        rows.append(
            {
                "id": row["id"],
                "retrieved_ids": retrieved,
                "reference_ids": reference,
                "context_precision": float(precision.single_turn_score(sample)) if retrieved and reference else None,
                "context_recall": float(recall.single_turn_score(sample)) if reference else None,
                "answer_verdict": row["verdict"],
            }
        )
    return rows


def metric_mean(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [row[field] for row in rows if row[field] is not None]
    return {"mean": mean(values) if values else None, "n": len(values)}


def evaluate(lab: Path) -> dict[str, Any]:
    questions = yaml.safe_load((lab / "configs" / "questions.yaml").read_text(encoding="utf-8"))["questions"]
    runs = []
    for graph in GRAPHS:
        units, refs = reference_units(lab.parent / "lab-2" / f"graphrag-{graph}" / "output", questions)
        for retriever in RETRIEVERS:
            path = lab / "results" / "answers" / f"{graph}_{retriever}.json"
            rows = score_run(path, graph, retriever, units, refs)
            runs.append(
                {
                    "graph": graph,
                    "retriever": retriever,
                    "questions": len(rows),
                    "indexed_units": len(units),
                    "questions_with_reference": sum(bool(ids) for ids in refs.values()),
                    "context_precision": metric_mean(rows, "context_precision"),
                    "context_recall": metric_mean(rows, "context_recall"),
                    "correct": sum(row["answer_verdict"] == "correct" for row in rows),
                    "rows": rows,
                }
            )
    return {"ragas_version": version("ragas"), "runs": runs}


@APP.callback()
def main() -> None:
    pass


@APP.command()
def run(lab: Path = LAB_OPTION, output: Path = OUTPUT_OPTION) -> None:
    try:
        report = evaluate(lab)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(output)
        logger.info("Wrote {} RAG comparisons to {}", len(report["runs"]), output)
    except Exception:
        logger.exception("Ragas evaluation failed.")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    APP()
