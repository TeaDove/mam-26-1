import gzip
import logging
import re
import statistics
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
    GoldPageMetrics,
    NormalizationMetrics,
    RetrievalMetrics,
    TokenizationMetrics,
    VectorizationMetrics,
)
from lab2.service.chunk_metrics import BookChunks, Span, chunking_metrics
from lab2.service.chunking import StructuralChunker
from lab2.service.cleaning import Cleaner, CleaningConfig
from lab2.service.cleaning_metrics import CleaningEvaluator, combine
from lab2.service.corpus import Corpus, strip_bibliography, token_windows
from lab2.service.figures import draw_background, draw_graph
from lab2.service.gold_eval import GoldEvaluator, load_pages
from lab2.service.graph_criteria import (
    duplicate_groups,
    embedding_duplicates,
    formula_atoms,
    formulas_and_tables,
    integrity,
    lemma_key,
    lift_pronouns,
    noise_labels,
    rule_counts,
    sample_nodes,
    source_vocabulary,
    table_atoms,
    text_terms,
    window,
)
from lab2.service.graph_eval import EvaluationConfig, coverage, load_graph, structure_metrics, traversal
from lab2.service.judge import check_consistency
from lab2.service.normalization import Glossary, Normalizer
from lab2.service.normalization_metrics import ANNOTATION_RULES, NormalizationEvaluator
from lab2.service.retrieval import QuerySet, RankedQuery, rank, relevant, summarize
from lab2.service.tokenization import FORMULA_CONVERSION_RULES, TextTokenizer
from lab2.service.vectorization import VectorDocuments, save_tfidf, vectorization_metrics, word_windows
from lab2.supplier.embeddings import EmbeddingClient
from lab2.supplier.llm import LlmClient
from lab2.util.io import read_jsonl, write_json, write_jsonl
from lab2.util.report import markdown_table
from lab2.util.settings import Settings
from lab2.util.text import WORD, count_tokens, language

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


def load_chunks(settings: Settings, name: str) -> list[Chunk]:
    return read_jsonl(settings.data_dir / name, Chunk)


def _join_book(chunks: list[Chunk], book: str) -> str:
    return "\n\n".join(chunk.text for chunk in chunks if chunk.book == book)


def _book_of(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def _book_chunk(book: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{book}_all",
        book=book,
        lang=language(text),
        section="",
        text=text,
        n_tokens=count_tokens(text),
        start=0,
        end=len(text),
    )


def run_prepare(settings: Settings) -> None:
    config = CleaningConfig.load(settings.cleaning_path)
    raw = Corpus(raw_dir=settings.raw_dir, books=settings.books)
    settings.dirty_dir.mkdir(parents=True, exist_ok=True)
    for book in settings.books:
        text = strip_bibliography(raw.raw_text(book), config.books[book].bibliography_start)
        (settings.dirty_dir / f"{book}.md").write_text(text, encoding="utf-8")
        log.info("%s: %d → %d characters without bibliography", book, len(raw.raw_text(book)), len(text))


def run_cleaning(settings: Settings) -> None:
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    config = CleaningConfig.load(settings.cleaning_path)
    raw_books = {book: corpus.raw_text(book) for book in settings.books}
    cleaner = Cleaner(config=config)
    cleaned = cleaner.clean([_book_chunk(book, text) for book, text in raw_books.items()], raw_books)
    write_jsonl(settings.data_dir / "01_clean.jsonl", cleaned)
    write_jsonl(settings.data_dir / "01_clean_audit.jsonl", cleaner.audit)
    evaluator = CleaningEvaluator(config=config)
    columns: dict[str, CleaningMetrics] = {}
    dirty_parts: list[CleaningMetrics] = []
    clean_parts: list[CleaningMetrics] = []
    for book in settings.books:
        audit = [entry for entry in cleaner.audit if _book_of(entry.chunk_id) == book]
        headers = cleaner.headers_for(raw_books[book])
        dirty, clean = evaluator.compare(raw_books[book], _join_book(cleaned, book), book, headers, audit)
        columns[f"dirty:{book}"], columns[f"clean:{book}"] = dirty, clean
        dirty_parts.append(dirty)
        clean_parts.append(clean)
    columns["dirty:total"] = combine(dirty_parts)
    columns["clean:total"] = combine(clean_parts)
    rules = Counter(entry.rule for entry in cleaner.audit)
    write_json(settings.metrics_dir / "01_cleaning_rules.json", dict(rules.most_common()))
    save_metrics(settings, "01_cleaning", "Stage 1: cleaning", columns)


def run_normalization(settings: Settings) -> None:
    glossary = Glossary.load(settings.glossary_path)
    cleaned = load_chunks(settings, "01_clean.jsonl")
    normalizer = Normalizer(glossary=glossary)
    normalized = normalizer.normalize(cleaned)
    audit_all = normalizer.audit + normalizer.lemmatizer.audit
    write_jsonl(settings.data_dir / "02_normalized.jsonl", normalized)
    write_jsonl(settings.data_dir / "02_normalized_audit.jsonl", audit_all)
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
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
        lemma_stats = [normalizer.lemmatizer.stats.get(f"{book}_all") for book in books]
        for stage, texts in stages.items():
            text = "\n\n".join(texts[book] for book in books)
            before = "\n\n".join(stages["cleaned"][book] for book in books) if stage == "normalized" else None
            metrics = evaluator.evaluate(text, before, audit if stage == "normalized" else [])
            if stage == "normalized":
                words = sum(s.words for s in lemma_stats if s)
                changed = sum(s.changed for s in lemma_stats if s)
                full = sum(s.full_changed for s in lemma_stats if s)
                metrics = metrics.model_copy(
                    update={
                        "lemmatized_words": changed,
                        "lemmatized_share": round(changed / max(words, 1), 4),
                        "full_lemmatization_share": round(full / max(words, 1), 4),
                    }
                )
            if scope == "total":
                metrics = metrics.model_copy(
                    update={"bridging_terms": evaluator.bridging_terms([texts[book] for book in books])}
                )
            columns[f"{stage}:{scope}"] = metrics
    rules = Counter(entry.rule for entry in audit_all)
    write_json(settings.metrics_dir / "02_normalization_rules.json", dict(rules.most_common()))
    save_metrics(settings, "02_normalization", "Stage 2: normalization", columns)


def run_chunking(settings: Settings) -> None:
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    normalized = {chunk.book: chunk.text for chunk in load_chunks(settings, "02_normalized.jsonl")}
    chunker = StructuralChunker(
        target_tokens=settings.chunk_target_tokens,
        min_tokens=settings.chunk_min_tokens,
        hard_max_tokens=settings.chunk_hard_max_tokens,
    )
    structural = [chunk for book in settings.books for chunk in chunker.chunk(book, normalized[book])]
    write_jsonl(settings.data_dir / "03_chunks.jsonl", structural)
    dirty = [
        unit
        for book in settings.books
        for unit in token_windows(
            book, corpus.raw_text(book), settings.graphrag_chunk_tokens, settings.graphrag_chunk_overlap
        )
    ]
    write_jsonl(settings.data_dir / "00_dirty_units.jsonl", dirty)
    arms = {"dirty_graphrag": (dirty, corpus.raw_text), "structural": (structural, normalized.__getitem__)}
    columns: dict[str, BaseModel] = {}
    for scope in (*settings.books, "total"):
        books = settings.books if scope == "total" else (scope,)
        for arm, (chunks, source) in arms.items():
            grouped = [
                BookChunks(
                    book_text=source(book),
                    spans=[Span(text=c.text, start=c.start, end=c.end) for c in chunks if c.book == book],
                )
                for book in books
            ]
            columns[f"{arm}:{scope}"] = chunking_metrics(grouped, settings.graphrag_chunk_tokens)
    save_metrics(settings, "03_chunking", "Stage 3: chunking", columns)


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
    audit = read_jsonl(settings.data_dir / "02_normalized_audit.jsonl", AuditEntry)
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
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    dirty_units = load_chunks(settings, "00_dirty_units.jsonl")
    cleaned = load_chunks(settings, "01_clean.jsonl")
    chunks = load_chunks(settings, "03_chunks.jsonl")
    write_jsonl(settings.data_dir / "04_tokens.jsonl", [tokenizer.tokenize(c) for c in chunks])
    write_jsonl(settings.data_dir / "04_tokens_dirty.jsonl", [tokenizer.tokenize(c) for c in dirty_units])
    effects = _normalization_effects(settings)
    columns: dict[str, TokenizationMetrics] = {}
    for scope in (*settings.books, "total"):
        books = settings.books if scope == "total" else (scope,)
        raw_texts = [corpus.raw_text(book) for book in books]
        units = [unit.text for unit in dirty_units if unit.book in books]
        columns[f"raw:{scope}"] = tokenizer.metrics(raw_texts, units, 0, 0)
        clean_texts = [c.text for c in cleaned if c.book in books]
        columns[f"cleaned:{scope}"] = tokenizer.metrics(clean_texts, clean_texts, 0, 0)
        chunk_texts = [c.text for c in chunks if c.book in books]
        inserted = sum(effects[book][0] for book in books)
        converted = sum(effects[book][1] for book in books)
        columns[f"normalized:{scope}"] = tokenizer.metrics(chunk_texts, chunk_texts, inserted, converted)
    save_metrics(settings, "04_tokenization", "Stage 4: tokenization", columns)


def run_vectorization(settings: Settings) -> None:
    client = EmbeddingClient(url=settings.embedding_url, model=settings.embedding_model)
    tokenizer = build_tokenizer(settings)
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    arms = {
        "dirty_units": load_chunks(settings, "00_dirty_units.jsonl"),
        "clean_chunks": load_chunks(settings, "03_chunks.jsonl"),
    }
    book_words = {
        "dirty_units": {book: tokenizer.words(corpus.raw_text(book)) for book in settings.books},
        "clean_chunks": {book: tokenizer.words(_join_book(arms["clean_chunks"], book)) for book in settings.books},
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
            if scope == "total" and arm == "clean_chunks":
                save_tfidf(docs, settings.data_dir / "05_tfidf_clean")
    save_metrics(settings, "05_vectorization", "Stage 5: vectorization", columns)


def run_export(settings: Settings) -> None:
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    chunks = load_chunks(settings, "03_chunks.jsonl")
    oversized = [chunk.chunk_id for chunk in chunks if chunk.n_tokens > settings.graphrag_chunk_tokens]
    if oversized:
        raise ValueError(f"chunks exceed the GraphRAG chunk size: {oversized}")
    documents = {
        "graphrag-clean": {f"{chunk.chunk_id}.md": chunk.text for chunk in chunks},
        "graphrag-dirty": {f"{book}.md": corpus.raw_text(book) for book in settings.books},
    }
    for workspace, files in documents.items():
        target = settings.root / workspace / "input"
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.glob("*.md"):
            stale.unlink()
        for name, text in files.items():
            (target / name).write_text(text.rstrip() + "\n", encoding="utf-8")
        log.info("exported %d documents to %s", len(files), target)


def _api_key(settings: Settings) -> str:
    api_key = settings.openai_api_key or sys.stdin.readline().strip()
    if not api_key:
        raise ValueError("OpenAI API key is empty: set LAB2_OPENAI_API_KEY or pass it on stdin")
    return api_key


def _source_fragment(node: str, units: list[str], unit_text: dict[str, str]) -> str:
    best = max(units, key=lambda unit: _mentions(node, str(unit_text.get(unit, ""))), default="")
    return str(unit_text.get(best, ""))


def _mentions(name: str, text: str) -> int:
    total = 0
    for word in re.findall(r"[^\W_]+", name):
        if len(word) < 4:
            total += len(re.findall(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])", text))
        else:
            stem = word[: max(4, len(word) - 2)]
            total += len(re.findall(rf"(?<![^\W_]){re.escape(stem)}", text, re.I))
    return total


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = left / np.linalg.norm(left, axis=1, keepdims=True)
    b = right / np.linalg.norm(right, axis=1, keepdims=True)
    return a @ b.T


def run_compare(settings: Settings) -> None:
    client = LlmClient(
        base_url=settings.openai_base_url,
        api_key=_api_key(settings),
        model=settings.judge_model,
        cache_dir=settings.data_dir / "judge_cache",
    )
    embedder = EmbeddingClient(url=settings.embedding_url, model=settings.embedding_model)
    config = EvaluationConfig.load(settings.evaluation_path)
    cleaned = {chunk.book: chunk.text for chunk in load_chunks(settings, "01_clean.jsonl")}
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    chunks = load_chunks(settings, "03_chunks.jsonl")
    sources = {
        "dirty": "\n\n".join(corpus.raw_text(book) for book in settings.books),
        "clean": "\n\n".join(chunk.text for chunk in chunks),
    }
    terms = text_terms(cleaned["tanaka1981"], "en", config.terms_en)
    terms += text_terms(cleaned["stat3"], "ru", config.terms_ru)
    glossary = Glossary.load(settings.glossary_path)
    vocabulary = source_vocabulary([*cleaned.values(), sources["clean"]])
    allowed = frozenset(
        {t.symbol.casefold() for t in glossary.terms if t.symbol}
        | {a.short.casefold() for a in glossary.abbreviations}
        | {"ar1", "ar3", "ac1", "ac3", "nb", "ti", "v", "mo", "mn", "si", "al", "cr", "ni", "cu"}
    )
    term_vectors = embedder.embed(terms)
    concepts = list(config.concepts)
    concept_vectors = embedder.embed([" / ".join(config.concepts[c]) for c in concepts])
    outputs = {
        "dirty": settings.root / settings.dirty_graphrag_dir,
        "clean": settings.root / settings.clean_graphrag_dir,
    }
    report: dict[str, object] = {"reference_terms": terms}
    for arm, output in outputs.items():
        bundle = load_graph(output)
        graph = bundle.graph
        input_tokens = int(bundle.artifacts.text_units["n_tokens"].sum())
        names = [str(n) for n in graph.nodes]
        title_vectors = embedder.embed(names)
        noise = {n: noise_labels(n, vocabulary, allowed, graph.nodes[n]["type"]) for n in names}

        term_similarity = _cosine(term_vectors, title_vectors)
        keys = {n: f" {lemma_key(n)} " for n in names}
        exact = [any(f" {lemma_key(t)} " in key for key in keys.values()) for t in terms]
        semantic = [bool(e or term_similarity[i].max() >= config.term_similarity) for i, e in enumerate(exact)]

        lemma_groups = duplicate_groups(names)
        embedded = embedding_duplicates(names, title_vectors, config.duplicate_similarity)
        cross = [p for p in embedded if {graph.nodes[p[0]]["lang"], graph.nodes[p[1]]["lang"]} == {"en", "ru"}]
        lifted, subtree = lift_pronouns(graph)

        formulas, tables = formulas_and_tables(sources[arm])
        formula_result = integrity(graph, [formula_atoms(f) for f in formulas])
        table_result = integrity(graph, [table_atoms(cells) for cells in tables])

        concept_similarity = _cosine(concept_vectors, title_vectors).max(axis=1)
        name_coverage = coverage(graph, config)
        vector_found = [c for c, s in zip(concepts, concept_similarity, strict=True) if s >= config.term_similarity]

        entity_units = {
            row.title: list(row.text_unit_ids)
            for row in bundle.artifacts.entities.itertuples()
            if len(row.text_unit_ids)
        }
        unit_text = dict(zip(bundle.artifacts.text_units["id"], bundle.artifacts.text_units["text"], strict=True))
        sample = sample_nodes(graph, list(name_coverage["resolved"].values()), config.consistency_sample)
        verdicts = [
            check_consistency(
                client,
                graph,
                node,
                window(graph, node),
                _source_fragment(node, entity_units.get(node, []), unit_text),
            )
            for node in sample
        ]
        noisy = [n for n, labels in noise.items() if labels]
        report[arm] = {
            "input": {
                "text_units": len(bundle.artifacts.text_units),
                "input_tokens": input_tokens,
                "vertices_per_1k_tokens": round(len(names) / input_tokens * 1000, 2),
                "noise_vertices_per_1k_tokens": round(len(noisy) / input_tokens * 1000, 2),
            },
            "structure": structure_metrics(graph),
            "completeness": {
                "reference_terms": len(terms),
                "covered_exact": sum(exact),
                "covered_exact_share": round(sum(exact) / max(len(terms), 1), 4),
                "covered_semantic": sum(semantic),
                "covered_semantic_share": round(sum(semantic) / max(len(terms), 1), 4),
                "missing_examples": [t for t, s in zip(terms, semantic, strict=True) if not s][:25],
            },
            "noise": {
                "vertices": len(names),
                "noise_vertices": len(noisy),
                "noise_share": round(len(noisy) / max(len(names), 1), 4),
                "by_rule": rule_counts(noise),
                "examples": sorted(noisy, key=str)[:40],
            },
            "coreference": {
                "lemma_duplicate_groups": len(lemma_groups),
                "lemma_duplicate_vertices": sum(len(g) for g in lemma_groups),
                "embedding_duplicate_pairs": len(embedded),
                "cross_language_pairs": len(cross),
                "pronoun_vertices_lifted": lifted,
                "pronoun_subtree_vertices": subtree,
                "lemma_examples": [" = ".join(g) for g in lemma_groups[:15]],
                "embedding_examples": [f"{a} = {b} ({s})" for a, b, s in embedded[:15]],
            },
            "integrity": {
                "formulas": len(formulas),
                "formulas_evaluated": formula_result.in_vertex + formula_result.in_window + formula_result.broken,
                "formulas_in_vertex": formula_result.in_vertex,
                "formulas_in_window": formula_result.in_window,
                "formulas_broken": formula_result.broken,
                "formula_mean_window_share": round(statistics.fmean(formula_result.best_share or [0.0]), 3),
                "tables": len(tables),
                "tables_evaluated": table_result.in_vertex + table_result.in_window + table_result.broken,
                "tables_in_vertex": table_result.in_vertex,
                "tables_in_window": table_result.in_window,
                "tables_broken": table_result.broken,
                "table_best_window_share": table_result.best_share,
            },
            "consistency": {
                "checked_vertices": len(verdicts),
                "fully_defined_share": round(
                    sum(v["definition"] == "full" for v in verdicts) / max(len(verdicts), 1), 4
                ),
                "partially_defined_share": round(
                    sum(v["definition"] == "partial" for v in verdicts) / max(len(verdicts), 1), 4
                ),
                "contradictions_total": sum(int(v["contradictions"]) for v in verdicts),
                "vertices_with_contradictions_share": round(
                    sum(int(v["contradictions"]) > 0 for v in verdicts) / max(len(verdicts), 1), 4
                ),
            },
            "coverage": {
                **name_coverage,
                "vector_found": len(vector_found),
                "vector_coverage": round(len(vector_found) / max(len(concepts), 1), 4),
            },
            "traversal": traversal(graph, config, {n: "noise" for n in noisy}),
        }
        write_json(settings.metrics_dir / f"08_consistency_{arm}.json", verdicts)
        log.info("%s evaluated; judge tokens: %d in, %d out", arm, client.prompt_tokens, client.completion_tokens)
    report["judge_usage"] = {
        "model": settings.judge_model,
        "calls": client.calls,
        "cached_calls": client.cached_calls,
        "request_tokens_all_calls": client.request_tokens,
        "new_calls_prompt_tokens": client.prompt_tokens,
        "new_calls_completion_tokens": client.completion_tokens,
    }
    write_json(settings.metrics_dir / "08_graph_comparison.json", report)
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


def run_gold(settings: Settings) -> None:
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    evaluator = GoldEvaluator(
        cleaning=CleaningConfig.load(settings.cleaning_path), glossary=Glossary.load(settings.glossary_path)
    )
    pages = load_pages(settings.data_dir / "gold" / "pages")
    cleaned = {chunk.book: chunk.text for chunk in load_chunks(settings, "01_clean.jsonl")}
    columns = {page.name: evaluator.evaluate(page, corpus.raw_text(page.book), cleaned[page.book]) for page in pages}
    total = _gold_total(list(columns.values()))
    save_metrics(settings, "06_gold", "Gold pages: cleaning and normalization quality", {**columns, "total": total})


def _gold_total(parts: list[GoldPageMetrics]) -> GoldPageMetrics:
    chars = sum(p.gold_characters for p in parts)
    should = sum(p.blocks_should_delete for p in parts)
    deleted = sum(p.blocks_deleted for p in parts)
    correct = sum(p.blocks_deleted_correctly for p in parts)
    units = sum(p.units_expected for p in parts)
    found = sum(p.units_correct for p in parts)
    terms = sum(p.terms_expected for p in parts)
    terms_found = sum(p.terms_correct for p in parts)
    precision = correct / deleted if deleted else 1.0
    recall = correct / should if should else 1.0

    def weighted(name: str) -> float:
        return round(sum(getattr(p, name) * p.gold_characters for p in parts) / max(chars, 1), 4)

    return GoldPageMetrics(
        gold_characters=chars,
        raw_cer=weighted("raw_cer"),
        clean_cer=weighted("clean_cer"),
        raw_wer=weighted("raw_wer"),
        clean_wer=weighted("clean_wer"),
        blocks=sum(p.blocks for p in parts),
        blocks_should_delete=should,
        blocks_deleted=deleted,
        blocks_deleted_correctly=correct,
        deletion_precision=round(precision, 4),
        deletion_recall=round(recall, 4),
        deletion_f1=round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        units_expected=units,
        units_correct=found,
        unit_accuracy=round(found / units, 4) if units else None,
        terms_expected=terms,
        terms_correct=terms_found,
        term_accuracy=round(terms_found / terms, 4) if terms else None,
    )


def run_retrieval(settings: Settings) -> None:
    client = EmbeddingClient(url=settings.embedding_url, model=settings.embedding_model)
    queries = QuerySet.load(settings.root / "configs" / "queries.yaml").queries
    corpus = Corpus(raw_dir=settings.dirty_dir, books=settings.books)
    normalized = {chunk.book: chunk.text for chunk in load_chunks(settings, "02_normalized.jsonl")}
    chunker = StructuralChunker(
        target_tokens=settings.chunk_target_tokens,
        min_tokens=settings.chunk_min_tokens,
        hard_max_tokens=settings.chunk_hard_max_tokens,
    )
    size, overlap = settings.graphrag_chunk_tokens, settings.graphrag_chunk_overlap
    arms = {
        "dirty_windows": load_chunks(settings, "00_dirty_units.jsonl"),
        "dirty_structural": [c for b in settings.books for c in chunker.chunk(b, corpus.raw_text(b))],
        "clean_windows": [c for b in settings.books for c in token_windows(b, normalized[b], size, overlap)],
        "clean_structural": load_chunks(settings, "03_chunks.jsonl"),
    }
    query_vectors = client.embed([query.question for query in queries])
    ranked: dict[str, list[RankedQuery]] = {}
    unanswerable: dict[str, list[str]] = {}
    for arm, chunks in arms.items():
        vectors = client.embed([chunk.text for chunk in chunks])
        order = rank(query_vectors, vectors)
        ranked[arm] = [
            RankedQuery(query=query, ranking=[int(i) for i in order[index]], relevant=relevant(query, chunks))
            for index, query in enumerate(queries)
        ]
        unanswerable[arm] = [item.query.id for item in ranked[arm] if not item.relevant]
    usable = {query.id for query in queries} - {q for ids in unanswerable.values() for q in ids}
    book_lang = {c.book: c.lang for c in arms["clean_structural"]}
    groups = {
        "all": lambda q: True,
        "same_language": lambda q: q.lang == book_lang[q.book],
        "cross_language": lambda q: q.lang != book_lang[q.book],
        "classmate_set": lambda q: q.source.startswith("classmate"),
    }
    columns: dict[str, RetrievalMetrics] = {}
    for group, keep in groups.items():
        for arm, items in ranked.items():
            selected = [item for item in items if item.query.id in usable and keep(item.query)]
            columns[f"{arm}:{group}"] = summarize(selected)
    write_json(
        settings.metrics_dir / "05_retrieval_details.json",
        {"unanswerable": unanswerable, "chunks": {arm: len(chunks) for arm, chunks in arms.items()}},
    )
    save_metrics(settings, "05_retrieval", "Stage 5: retrieval quality (bge-m3), chunking × preprocessing", columns)


def run_graphs(settings: Settings) -> None:
    embedder = EmbeddingClient(url=settings.embedding_url, model=settings.embedding_model)
    outputs = {
        "dirty": settings.root / settings.dirty_graphrag_dir,
        "clean": settings.root / settings.clean_graphrag_dir,
    }
    graph_dir = settings.root / "results" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}
    for arm, output in outputs.items():
        bundle = load_graph(output)
        graph = bundle.graph
        input_tokens = int(bundle.artifacts.text_units["n_tokens"].sum())
        names = [str(n) for n in graph.nodes]
        vectors = embedder.embed([f"{n}: {graph.nodes[n]['description'][:400]}" for n in names])
        for name, vector in zip(names, vectors, strict=True):
            graph.nodes[name]["embedding"] = " ".join(f"{value:.5f}" for value in vector)
        with gzip.open(graph_dir / f"{arm}_graph_with_vectors.graphml.gz", "wb") as stream:
            nx.write_graphml(graph, stream)
        summary[arm] = {
            "documents": len(bundle.artifacts.documents),
            "text_units": len(bundle.artifacts.text_units),
            "entities": len(bundle.artifacts.entities),
            "relationships": len(bundle.artifacts.relationships),
            "vertices": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "vertices_with_vectors": sum(1 for _, d in graph.nodes(data=True) if d.get("embedding")),
            "vector_dimensions": int(vectors.shape[1]),
            "reverse_relationships_merged": sum(d.get("merged", 1) - 1 for *_, d in graph.edges(data=True)),
            "input_tokens": input_tokens,
            "mean_text_unit_tokens": round(input_tokens / len(bundle.artifacts.text_units), 1),
            "vertices_per_1k_tokens": round(graph.number_of_nodes() / input_tokens * 1000, 2),
            "edges_per_1k_tokens": round(graph.number_of_edges() / input_tokens * 1000, 2),
        }
    write_json(settings.metrics_dir / "07_graphs.json", summary)
    log.info("graphs exported to %s", graph_dir)
