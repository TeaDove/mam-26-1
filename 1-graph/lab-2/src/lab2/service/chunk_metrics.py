import statistics
from dataclasses import dataclass
from itertools import pairwise

from lab2.dto.models import Block, BlockKind, ChunkingMetrics, TextStats
from lab2.util.text import (
    DISPLAY_MATH,
    INLINE_MATH,
    clean_cut,
    count_tokens,
    ends_sentence_strict,
    furniture_lines,
    is_citation,
    parse_blocks,
)


@dataclass
class Span:
    text: str
    start: int
    end: int


@dataclass
class BookChunks:
    book_text: str
    spans: list[Span]


def text_stats(values: list[int]) -> TextStats:
    return TextStats(
        count=len(values),
        mean=round(statistics.fmean(values), 1),
        median=float(statistics.median(values)),
        min=min(values),
        max=max(values),
        std=round(statistics.pstdev(values), 1),
    )


def chunking_metrics(books: list[BookChunks], graphrag_limit: int) -> ChunkingMetrics:
    token_counts: list[int] = []
    boundaries = cut_boundaries = end_mid = start_mid = 0
    formulas_total = formulas_broken = formulas_lost = 0
    tables_total = tables_broken = tables_lost = 0
    for book in books:
        blocks = parse_blocks(book.book_text)
        furniture = furniture_lines(book.book_text)
        formula_spans = _formula_spans(book.book_text)
        table_spans = [(b.start, b.end) for b in blocks if b.kind is BlockKind.TABLE]
        positions: set[int] = set()
        token_counts.extend(count_tokens(span.text) for span in book.spans)
        for left, right in pairwise(book.spans):
            ends_bad = _cuts_sentence(book.book_text, blocks, left.end, furniture)
            starts_bad = _cuts_sentence(book.book_text, blocks, right.start, furniture)
            overlapping = right.start < left.end
            boundaries += 1
            end_mid += ends_bad
            start_mid += starts_bad if overlapping else ends_bad
            cut_boundaries += ends_bad or (overlapping and starts_bad)
            positions.update({left.end, right.start} if overlapping else {left.end})
        formulas_total += len(formula_spans)
        tables_total += len(table_spans)
        formulas_broken += _count_cut(formula_spans, positions)
        tables_broken += _count_cut(table_spans, positions)
        formulas_lost += _count_not_whole(formula_spans, book.spans)
        tables_lost += _count_not_whole(table_spans, book.spans)
    return ChunkingMetrics(
        n_chunks=len(token_counts),
        tokens=text_stats(token_counts),
        end_mid_sentence_share=round(end_mid / max(boundaries, 1), 3),
        start_mid_sentence_share=round(start_mid / max(boundaries, 1), 3),
        boundaries=boundaries,
        boundaries_cutting_sentence_share=round(cut_boundaries / max(boundaries, 1), 3),
        formulas_total=formulas_total,
        formulas_broken=formulas_broken,
        formulas_broken_share=round(formulas_broken / max(formulas_total, 1), 4),
        formulas_not_whole_anywhere=formulas_lost,
        tables_total=tables_total,
        tables_broken=tables_broken,
        tables_not_whole_anywhere=tables_lost,
        over_graphrag_limit=sum(1 for count in token_counts if count > graphrag_limit),
    )


def _count_cut(spans: list[tuple[int, int]], positions: set[int]) -> int:
    return sum(1 for start, end in spans if any(start < p < end for p in positions))


def _count_not_whole(spans: list[tuple[int, int]], chunks: list[Span]) -> int:
    return sum(1 for start, end in spans if not any(c.start <= start and end <= c.end for c in chunks))


def _formula_spans(text: str) -> list[tuple[int, int]]:
    display = [(m.start(), m.end()) for m in DISPLAY_MATH.finditer(text)]
    masked = DISPLAY_MATH.sub(lambda m: " " * len(m.group()), text)
    inline = [(m.start(), m.end()) for m in INLINE_MATH.finditer(masked)]
    return display + [(s, e) for s, e in inline if not is_citation(text[s:e])]


def _cuts_sentence(text: str, blocks: list[Block], position: int, furniture: frozenset[str]) -> bool:
    for block in blocks:
        if block.start < position < block.end:
            if block.kind is not BlockKind.PARAGRAPH:
                return True
            before = text[block.start : position].rstrip()
            return not ends_sentence_strict(before) or text[position - 1].isalnum()
    left = [b for b in blocks if b.end <= position]
    right = [b for b in blocks if b.start >= position]
    return not clean_cut(left, right, furniture)
