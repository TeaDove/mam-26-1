import logging
import re
import sys
from collections import Counter
from collections.abc import Callable

import networkx as nx
import numpy as np
from pydantic import BaseModel

from lab2.dto.models import (
    AuditEntry,
    Chunk,
    CleaningMetrics,
    NormalizationMetrics,
    TokenizationMetrics,
    VectorizationMetrics,
)
from lab2.service.chunk_metrics import BookChunks, Span, chunking_metrics
from lab2.service.chunking import StructuralChunker
from lab2.service.cleaning import Cleaner, CleaningConfig
from lab2.service.cleaning_metrics import CleaningEvaluator, combine
from lab2.service.corpus import Corpus
from lab2.service.figures import draw_background, draw_graph
from lab2.service.graph_eval import EvaluationConfig, coverage, load_graph, structure_metrics, traversal
from lab2.service.judge import Judge, label_summary, summarize_units
from lab2.service.normalization import Glossary, Normalizer
from lab2.service.normalization_metrics import ANNOTATION_RULES, NormalizationEvaluator
from lab2.service.tokenization import FORMULA_CONVERSION_RULES, TextTokenizer
from lab2.service.vectorization import VectorDocuments, save_tfidf, vectorization_metrics, word_windows
from lab2.supplier.embeddings import EmbeddingClient
from lab2.supplier.llm import LlmClient
from lab2.util.io import read_jsonl, write_json, write_jsonl
from lab2.util.report import markdown_table
from lab2.util.settings import Settings
from lab2.util.text import WORD

log = logging.getLogger(__name__)


def per_book_columns[M: BaseModel](
    books: tuple[str, ...],
    arms: dict[str, list[Chunk]],
    compute: Callable[[list[Chunk]], M],
) -> dict[str, M]:
    columns: dict[str, M] = {}
    for scope in (*books, "total"):
        for arm, chunks in arms.items():
            selected = chunks if scope == "total" else [c for c in chunks if c.book == scope]
            columns[f"{arm}:{scope}"] = compute(selected)
    return columns


def save_metrics(settings: Settings, name: str, title: str, columns: dict[str, BaseModel]) -> None:
    write_json(settings.metrics_dir / f"{name}.json", {key: value for key, value in columns.items()})
    (settings.metrics_dir / f"{name}.md").write_text(markdown_table(title, columns), encoding="utf-8")
    log.info("metrics written: %s", settings.metrics_dir / f"{name}.md")


def run_chunking(settings: Settings) -> None:
    corpus = Corpus(raw_dir=settings.raw_dir, books=settings.books)
    chunker = StructuralChunker(
        target_tokens=settings.chunk_target_tokens,
        min_tokens=settings.chunk_min_tokens,
        hard_max_tokens=settings.chunk_hard_max_tokens,
    )
    structural = [chunk for book in settings.books for chunk in chunker.chunk(book, corpus.raw_text(book))]
    write_jsonl(settings.data_dir / "01_chunks.jsonl", structural)
    dirty = corpus.dirty_chunks(settings.root / settings.dirty_graphrag_dir)
    write_jsonl(settings.data_dir / "00_dirty_text_units.jsonl", dirty)

    def compute(chunks: list[Chunk]) -> BaseModel:
        grouped = [
            BookChunks(
                book_text=corpus.raw_text(book),
                spans=[Span(text=c.text, start=c.start, end=c.end) for c in chunks if c.book == book],
            )
            for book in settings.books
            if any(c.book == book for c in chunks)
        ]
        return chunking_metrics(grouped, settings.graphrag_chunk_tokens)

    columns = per_book_columns(settings.books, {"dirty_graphrag": dirty, "structural": structural}, compute)
    save_metrics(settings, "01_chunking", "Stage 1: chunking", columns)


def load_chunks(settings: Settings, name: str) -> list[Chunk]:
    return read_jsonl(settings.data_dir / name, Chunk)


def run_cleaning(settings: Settings) -> None:
    corpus = Corpus(raw_dir=settings.raw_dir, books=settings.books)
    config = CleaningConfig.load(settings.cleaning_path)
    raw_books = {book: corpus.raw_text(book) for book in settings.books}
    cleaner = Cleaner(config=config)
    cleaned = cleaner.clean(load_chunks(settings, "01_chunks.jsonl"), raw_books)
    write_jsonl(settings.data_dir / "02_clean.jsonl", cleaned)
    write_jsonl(settings.data_dir / "02_clean_audit.jsonl", cleaner.audit)
    evaluator = CleaningEvaluator(config=config)
    columns: dict[str, CleaningMetrics] = {}
    dirty_parts: list[CleaningMetrics] = []
    clean_parts: list[CleaningMetrics] = []
    for book in settings.books:
        prefix = f"{book}_c"
        audit = [entry for entry in cleaner.audit if entry.chunk_id.startswith(prefix)]
        clean_text = "\n\n".join(chunk.text for chunk in cleaned if chunk.book == book)
        headers = cleaner.headers_for(raw_books[book])
        dirty, clean = evaluator.compare(raw_books[book], clean_text, book, headers, audit)
        columns[f"dirty:{book}"], columns[f"clean:{book}"] = dirty, clean
        dirty_parts.append(dirty)
        clean_parts.append(clean)
    columns["dirty:total"] = combine(dirty_parts)
    columns["clean:total"] = combine(clean_parts)
    rules = Counter(entry.rule for entry in cleaner.audit)
    write_json(settings.metrics_dir / "02_cleaning_rules.json", dict(rules.most_common()))
    save_metrics(settings, "02_cleaning", "Stage 2: cleaning", columns)


def run_normalization(settings: Settings) -> None:
    glossary = Glossary.load(settings.glossary_path)
    cleaned = load_chunks(settings, "02_clean.jsonl")
    normalizer = Normalizer(glossary=glossary)
    normalized = normalizer.normalize(cleaned)
    write_jsonl(settings.data_dir / "03_normalized.jsonl", normalized)
    write_jsonl(settings.data_dir / "03_normalized_audit.jsonl", normalizer.audit)
    corpus = Corpus(raw_dir=settings.raw_dir, books=settings.books)
    evaluator = NormalizationEvaluator(glossary=glossary)
    stages = {
        "raw": {book: corpus.raw_text(book) for book in settings.books},
        "cleaned": {book: _join_book(cleaned, book) for book in settings.books},
        "normalized": {book: _join_book(normalized, book) for book in settings.books},
    }
    columns: dict[str, NormalizationMetrics] = {}
    for scope in (*settings.books, "total"):
        books = settings.books if scope == "total" else (scope,)
        audit = [entry for entry in normalizer.audit if _book_of(entry.chunk_id) in books]
        for stage, texts in stages.items():
            text = "\n\n".join(texts[book] for book in books)
            before = "\n\n".join(stages["cleaned"][book] for book in books) if stage == "normalized" else None
            metrics = evaluator.evaluate(text, before, audit if stage == "normalized" else [])
            if scope == "total":
                metrics = metrics.model_copy(
                    update={"bridging_terms": evaluator.bridging_terms([texts[book] for book in books])}
                )
            columns[f"{stage}:{scope}"] = metrics
    rules = Counter(entry.rule for entry in normalizer.audit)
    write_json(settings.metrics_dir / "03_normalization_rules.json", dict(rules.most_common()))
    save_metrics(settings, "03_normalization", "Stage 3: normalization", columns)


def _join_book(chunks: list[Chunk], book: str) -> str:
    return "\n\n".join(chunk.text for chunk in chunks if chunk.book == book)


def _book_of(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def build_tokenizer(settings: Settings) -> TextTokenizer:
    glossary = Glossary.load(settings.glossary_path)
    lexicon: set[str] = set()
    for term in glossary.terms:
        lexicon.update(term.en.casefold().replace("–", " ").replace("+", " ").split())
        if term.symbol:
            lexicon.add(term.symbol.casefold())
    for abbreviation in glossary.abbreviations:
        lexicon.add(abbreviation.short.casefold())
        lexicon.update(abbreviation.en.casefold().split())
    patterns = tuple(re.compile(term.ru, re.I) for term in glossary.terms)
    return TextTokenizer(
        bge_tokenizer_path=settings.bge_tokenizer_path, domain_lexicon=frozenset(lexicon), domain_patterns=patterns
    )


def _normalization_effects(settings: Settings) -> dict[str, tuple[int, int]]:
    audit = read_jsonl(settings.data_dir / "03_normalized_audit.jsonl", AuditEntry)
    effects: dict[str, tuple[int, int]] = {}
    for book in settings.books:
        entries = [entry for entry in audit if _book_of(entry.chunk_id) == book]
        inserted = sum(
            len(WORD.findall(entry.after)) - len(WORD.findall(entry.before))
            for entry in entries
            if entry.rule in ANNOTATION_RULES
        )
        converted = sum(1 for entry in entries if entry.rule in FORMULA_CONVERSION_RULES)
        effects[book] = (inserted, converted)
    return effects


def run_tokenization(settings: Settings) -> None:
    tokenizer = build_tokenizer(settings)
    corpus = Corpus(raw_dir=settings.raw_dir, books=settings.books)
    dirty_units = load_chunks(settings, "00_dirty_text_units.jsonl")
    arms = {
        "cleaned": load_chunks(settings, "02_clean.jsonl"),
        "normalized": load_chunks(settings, "03_normalized.jsonl"),
    }
    write_jsonl(settings.data_dir / "04_tokens.jsonl", [tokenizer.tokenize(c) for c in arms["normalized"]])
    write_jsonl(settings.data_dir / "04_tokens_dirty.jsonl", [tokenizer.tokenize(c) for c in dirty_units])
    effects = _normalization_effects(settings)
    columns: dict[str, TokenizationMetrics] = {}
    for scope in (*settings.books, "total"):
        books = settings.books if scope == "total" else (scope,)
        raw_texts = [corpus.raw_text(book) for book in books]
        units = [unit.text for unit in dirty_units if unit.book in books]
        columns[f"raw:{scope}"] = tokenizer.metrics(raw_texts, units, 0, 0)
        for arm, chunks in arms.items():
            texts = [chunk.text for chunk in chunks if chunk.book in books]
            inserted = sum(effects[book][0] for book in books) if arm == "normalized" else 0
            converted = sum(effects[book][1] for book in books) if arm == "normalized" else 0
            columns[f"{arm}:{scope}"] = tokenizer.metrics(texts, texts, inserted, converted)
    save_metrics(settings, "04_tokenization", "Stage 4: tokenization", columns)


def run_vectorization(settings: Settings) -> None:
    client = EmbeddingClient(url=settings.embedding_url, model=settings.embedding_model)
    tokenizer = build_tokenizer(settings)
    corpus = Corpus(raw_dir=settings.raw_dir, books=settings.books)
    arms = {
        "dirty_units": load_chunks(settings, "00_dirty_text_units.jsonl"),
        "cleaned": load_chunks(settings, "02_clean.jsonl"),
        "normalized": load_chunks(settings, "03_normalized.jsonl"),
    }
    book_words = {
        "dirty_units": {book: tokenizer.words(corpus.raw_text(book)) for book in settings.books},
        "cleaned": {book: tokenizer.words(_join_book(arms["cleaned"], book)) for book in settings.books},
        "normalized": {book: tokenizer.words(_join_book(arms["normalized"], book)) for book in settings.books},
    }
    dense: dict[str, np.ndarray] = {}
    for arm, chunks in arms.items():
        dense[arm] = client.embed([chunk.text for chunk in chunks])
        np.save(settings.data_dir / f"05_dense_{arm}.npy", dense[arm])
        write_json(settings.data_dir / f"05_dense_{arm}_ids.json", [chunk.chunk_id for chunk in chunks])
    columns: dict[str, VectorizationMetrics] = {}
    for scope in (*settings.books, "total"):
        books = settings.books if scope == "total" else (scope,)
        for arm, chunks in arms.items():
            keep = [i for i, chunk in enumerate(chunks) if chunk.book in books]
            docs = VectorDocuments(
                ids=[chunks[i].chunk_id for i in keep],
                langs=[chunks[i].lang for i in keep],
                windows=[w for book in books for w in word_windows(book_words[arm][book], settings.tfidf_window)],
                dense=dense[arm][keep],
            )
            columns[f"{arm}:{scope}"] = vectorization_metrics(docs)
            if scope == "total" and arm == "normalized":
                save_tfidf(docs, settings.data_dir / "05_tfidf_normalized")
    save_metrics(settings, "05_vectorization", "Stage 5: vectorization", columns)


def run_export(settings: Settings) -> None:
    target = settings.root / "graphrag-clean" / "input"
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.md"):
        stale.unlink()
    chunks = load_chunks(settings, "03_normalized.jsonl")
    oversized = [chunk.chunk_id for chunk in chunks if chunk.n_tokens > settings.graphrag_chunk_tokens]
    if oversized:
        raise ValueError(f"chunks exceed the GraphRAG chunk size: {oversized}")
    for chunk in chunks:
        (target / f"{chunk.chunk_id}.md").write_text(chunk.text + "\n", encoding="utf-8")
    log.info("exported %d chunks to %s", len(chunks), target)


def run_compare(settings: Settings) -> None:
    api_key = settings.openai_api_key or sys.stdin.readline().strip()
    if not api_key:
        raise ValueError("OpenAI API key is empty: set LAB2_OPENAI_API_KEY or pass it on stdin")
    client = LlmClient(
        base_url=settings.openai_base_url,
        api_key=api_key,
        model=settings.judge_model,
        cache_dir=settings.data_dir / "judge_cache",
    )
    judge = Judge(client=client)
    embedder = EmbeddingClient(url=settings.embedding_url, model=settings.embedding_model)
    config = EvaluationConfig.load(settings.evaluation_path)
    outputs = {
        "dirty": settings.root / settings.dirty_graphrag_dir,
        "clean": settings.root / settings.clean_graphrag_dir,
    }
    report: dict[str, dict[str, object]] = {}
    for arm, output in outputs.items():
        bundle = load_graph(output)
        graph = bundle.graph
        names = list(graph.nodes)
        title_vectors = embedder.embed([str(name) for name in names])
        node_vectors = embedder.embed([f"{name}: {graph.nodes[name]['description'][:400]}" for name in names])
        labels = judge.label_nodes(graph)
        for name, vector in zip(names, node_vectors, strict=True):
            graph.nodes[name]["embedding"] = " ".join(f"{value:.5f}" for value in vector)
            graph.nodes[name]["judge_label"] = labels.get(name, "unlabeled")
        graph_dir = settings.root / "results" / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, graph_dir / f"{arm}_graph_with_vectors.graphml")
        np.save(graph_dir / f"{arm}_node_vectors.npy", node_vectors)
        write_json(graph_dir / f"{arm}_node_ids.json", [str(name) for name in names])
        units = judge.judge_units(bundle)
        write_json(settings.metrics_dir / f"07_judge_units_{arm}.json", units)
        duplicates = judge.duplicates(graph, dict(zip(names, title_vectors, strict=True)), config.duplicate_similarity)
        report[arm] = {
            "structure": structure_metrics(graph),
            "judge_nodes": label_summary(labels),
            "judge_units": summarize_units(units),
            "judge_units_by_book": {
                book: summarize_units([u for u in units if u["book"] == book]) for book in settings.books
            },
            "duplicates": duplicates,
            "coverage": coverage(graph, config),
            "traversal": traversal(graph, config, labels),
            "noise_examples": sorted((n for n, label in labels.items() if label == "noise"), key=str)[:40],
        }
        log.info(
            "%s evaluated; judge usage so far: %d in / %d out tokens",
            arm,
            client.prompt_tokens,
            client.completion_tokens,
        )
    report["judge_usage"] = {
        "model": settings.judge_model,
        "prompt_tokens": client.prompt_tokens,
        "completion_tokens": client.completion_tokens,
    }
    write_json(settings.metrics_dir / "07_graph_comparison.json", report)
    log.info("comparison written")


def run_figures(settings: Settings) -> None:
    target = settings.root / "results" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dirty": settings.root / settings.dirty_graphrag_dir,
        "clean": settings.root / settings.clean_graphrag_dir,
    }
    titles = {"dirty": "Грязный граф: текст MinerU как есть", "clean": "Чистый граф: после предобработки"}
    for arm, output in outputs.items():
        graph = load_graph(output).graph
        graph.remove_nodes_from([n for n, d in dict(graph.degree()).items() if d == 0])
        draw_graph(graph, titles[arm], target / f"{arm}_graph.png")
        if arm == "clean":
            draw_background(graph, target / "clean_graph_background.png")
    log.info("figures written to %s", target)
