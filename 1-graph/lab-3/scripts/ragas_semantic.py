import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

from lab3.answer import NO_ANSWER, context_text
from lab3.graph import load_graph

GRAPHS = ("dirty", "clean")
RETRIEVERS = ("bge-m3", "hybrid")
APP = typer.Typer()
LAB_OPTION = typer.Option(..., exists=True, file_okay=False)
OUTPUT_OPTION = typer.Option(...)


def pending_rows(lab: Path, done: set[tuple[str, str, str]]) -> list[tuple[dict[str, Any], str]]:
    rows = []
    for graph in GRAPHS:
        kg = load_graph(graph, lab.parent / "lab-2" / f"graphrag-{graph}" / "output")
        for retriever in RETRIEVERS:
            path = lab / "results" / "answers" / f"{graph}_{retriever}.json"
            for row in json.loads(path.read_text(encoding="utf-8")):
                if (graph, retriever, row["id"]) not in done:
                    rows.append((row, context_text(kg, row["context"], row["source_units"])))
    return rows


def valid_score(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    if not 0 <= value <= 1:
        raise ValueError(f"Score is outside the unit interval: {value}")
    return value


async def evaluate(lab: Path, output: Path, base_url: str, model: str, workers: int) -> None:
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
    if any(row["judge"] != model for row in rows):
        raise ValueError("The output contains scores from a different judge.")
    done = {(row["graph"], row["retriever"], row["id"]) for row in rows}
    jobs = pending_rows(lab, done)
    semaphore = asyncio.Semaphore(workers)
    async with AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=180.0) as client:
        llm = llm_factory(model, provider="openai", client=client, temperature=0, max_tokens=1536)
        recall = ContextRecall(llm=llm)
        faithfulness = Faithfulness(llm=llm)

        async def score(row: dict[str, Any], context: str) -> dict[str, Any]:
            question, answer = row["question"], row["answer"]
            abstention = answer.strip().casefold().rstrip(".!") == NO_ANSWER.casefold().rstrip(".!")
            async with semaphore:
                result = await recall.ascore(
                    user_input=question, retrieved_contexts=[context], reference=row["reference"]
                )
                grounding = None
                if not abstention:
                    faithful = await faithfulness.ascore(
                        user_input=question, response=answer, retrieved_contexts=[context]
                    )
                    grounding = valid_score(faithful.value)
            return {
                "graph": row["graph"],
                "retriever": row["retriever"],
                "id": row["id"],
                "judge": model,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "context_recall": valid_score(result.value),
                "faithfulness": grounding,
                "abstention": abstention,
            }

        with output.open("a", encoding="utf-8") as stream:
            for completed in asyncio.as_completed([score(row, context) for row, context in jobs]):
                result = await completed
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                logger.info("{} {} {} scored", result["graph"], result["retriever"], result["id"])


@APP.callback()
def main() -> None:
    pass


@APP.command()
def run(
    lab: Path = LAB_OPTION,
    output: Path = OUTPUT_OPTION,
    base_url: str = typer.Option(...),
    model: str = typer.Option(...),
    workers: int = typer.Option(2, min=1),
) -> None:
    try:
        asyncio.run(evaluate(lab, output, base_url, model, workers))
    except Exception:
        logger.exception("Ragas semantic evaluation failed.")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    APP()
