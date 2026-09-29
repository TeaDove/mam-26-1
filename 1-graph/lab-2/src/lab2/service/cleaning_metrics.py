import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

from lab2.dto.models import AuditEntry, BlockKind, CleaningMetrics
from lab2.service.cleaning import DELETION_RULES, MIXED_CITATION, CleaningConfig
from lab2.util.text import (
    CITATION,
    DISPLAY_MATH,
    INLINE_MATH,
    NUMBER,
    TABLE,
    formulas,
    is_caption,
    is_continuation,
    is_furniture,
    parse_blocks,
    table_cells,
)

NOISE_PATTERNS = [
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),
    re.compile(r"<small>.*?</small>", re.S),
    re.compile(r"</?(?:span|small|strong|u|b|i|em|sup|sub)[^>]*>"),
    re.compile(r"\\[*~]"),
    re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b-\u200f\ufeff\ufffd]"),
    re.compile(r"[\u4e00-\u9fff]"),
    re.compile(r"\\quad\s*\.(?:\s*\\quad\s*\.)+"),
    re.compile(r"(?<=\$)[^$\n]*?\d (?=\d)[^$\n]*?(?=\$)"),
    CITATION,
    re.compile(r"\[\d+(?:\s*[,–\-]\s*\d+)*\]"),
    re.compile(r"et al\.\d+"),
    re.compile(r"\bIMR/\d+"),
]
PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"(?:\+\d|\(\d{3,5}\))[\d\s()\-]{7,}\d"),
    re.compile(r"\b[А-ЯA-Z]\.\s?[А-ЯA-Z]\.\s?[А-ЯЁA-Z][а-яёa-z]+"),
    re.compile(r"is with the|рекомендована к печати|д\.т\.н\."),
]
HYPHEN_BREAK = re.compile(r"[^\W\d_]-(?:[ \t]+|\n\n)[a-zа-яё]")


@dataclass
class CleaningEvaluator:
    config: CleaningConfig

    def compare(
        self, dirty: str, clean: str, book: str, headers: frozenset[str], audit: list[AuditEntry]
    ) -> tuple[CleaningMetrics, CleaningMetrics]:
        deleted = [entry for entry in audit if entry.rule in DELETION_RULES]
        removed_numbers: Counter[str] = Counter()
        removed_formulas: Counter[str] = Counter()
        for entry in deleted:
            removed_numbers += _numbers(entry.before)
            removed_formulas += Counter(_canonical_formula(f) for f in formulas(entry.before))
        corrected = dirty
        for fix in self.config.ocr_fixes:
            corrected = corrected.replace(fix.source, fix.target)
        corrected = MIXED_CITATION.sub("", corrected)
        expected_numbers = _numbers(corrected) - removed_numbers
        expected_formulas = Counter(_canonical_formula(f) for f in formulas(corrected)) - removed_formulas
        clean_numbers = _numbers(clean)
        clean_formulas = Counter(_canonical_formula(f) for f in formulas(clean))
        lost_numbers = sum((expected_numbers - clean_numbers).values())
        kept_formulas = sum((expected_formulas & clean_formulas).values())
        dirty_cells = Counter(_canonical_cell(c) for c in table_cells(dirty))
        clean_cells = Counter(_canonical_cell(c) for c in table_cells(clean))
        dirty_metrics = self._metrics(dirty, book, headers)
        clean_metrics = self._metrics(clean, book, headers).model_copy(
            update={
                "numbers_removed_by_deletion": sum(removed_numbers.values()),
                "numbers_expected": sum(expected_numbers.values()),
                "numbers_lost": lost_numbers,
                "numbers_preserved_share": _share(sum(expected_numbers.values()) - lost_numbers, expected_numbers),
                "formulas_removed_by_deletion": sum(removed_formulas.values()),
                "formulas_preserved_share": _share(kept_formulas, expected_formulas),
                "table_cells_preserved_share": _share(sum((dirty_cells & clean_cells).values()), dirty_cells),
                "table_cells_modified": sum(1 for entry in audit if entry.rule == "table_cell_hyphen"),
                "ocr_corrections": sum(1 for entry in audit if entry.rule == "ocr_fix"),
            }
        )
        return dirty_metrics, clean_metrics

    def _metrics(self, text: str, book: str, headers: frozenset[str]) -> CleaningMetrics:
        noise = self._noise_characters(text, headers)
        numbers = sum(_numbers(text).values())
        return CleaningMetrics(
            characters=len(text),
            numbers=numbers,
            numbers_removed_by_deletion=0,
            numbers_expected=numbers,
            numbers_lost=0,
            numbers_preserved_share=1.0,
            formulas=len(formulas(text)),
            formulas_removed_by_deletion=0,
            formulas_preserved_share=1.0,
            tables=len(TABLE.findall(text)),
            table_cells=len(table_cells(text)),
            table_cells_preserved_share=1.0,
            table_cells_modified=0,
            noise_characters=noise,
            noise_share=round(noise / max(len(text), 1), 4),
            service_marks=self._service_marks(text, book, headers),
            pii_hits=sum(len(pattern.findall(text)) for pattern in PII_PATTERNS),
            broken_paragraphs=_broken_paragraphs(text, headers),
            hyphenated_breaks=len(HYPHEN_BREAK.findall(text)),
            ocr_corrections=0,
        )

    def _noise_characters(self, text: str, headers: frozenset[str]) -> int:
        body = TABLE.sub("", text)
        mask = bytearray(len(body))
        for pattern in NOISE_PATTERNS:
            for match in pattern.finditer(body):
                mask[match.start() : match.end()] = b"\x01" * (match.end() - match.start())
        for block in parse_blocks(body):
            if is_furniture(block, headers):
                mask[block.start : block.end] = b"\x01" * (block.end - block.start)
        return sum(mask)

    def _service_marks(self, text: str, book: str, headers: frozenset[str]) -> int:
        rules = self.config.books[book]
        marks = len(re.findall(r"!\[[^\]]*\]\(doc:", text)) + text.count("docvortex")
        marks += len(CITATION.findall(text)) + len(re.findall(r"\[\d+(?:\s*[,–\-]\s*\d+)*\]", text))
        marks += len(re.findall(r"et al\.\d+|\bIMR/\d+|\bУДК\b", text))
        for block in parse_blocks(text):
            stripped = block.text.strip()
            if is_furniture(block, headers) or any(re.match(p, stripped) for p in rules.drop_blocks):
                marks += 1
        return marks


def _broken_paragraphs(text: str, headers: frozenset[str]) -> int:
    body = [
        block
        for block in parse_blocks(text)
        if block.kind in (BlockKind.PARAGRAPH, BlockKind.HEADING, BlockKind.FORMULA, BlockKind.TABLE)
        and not is_furniture(block, headers)
        and not is_caption(block)
    ]
    pairs = sum(
        1
        for left, right in pairwise(body)
        if left.kind is BlockKind.PARAGRAPH
        and right.kind is BlockKind.PARAGRAPH
        and is_continuation(left.text, right.text)
    )
    dangling = sum(1 for block in body if block.kind is BlockKind.PARAGRAPH and block.text.rstrip().endswith("-"))
    return pairs + dangling


def _numbers(text: str) -> Counter[str]:
    def squash(match: re.Match[str]) -> str:
        return re.sub(r"(?:\\quad|\s)+", "", match.group(0))

    squashed = DISPLAY_MATH.sub(squash, text)
    squashed = INLINE_MATH.sub(squash, squashed)
    return Counter(NUMBER.findall(CITATION.sub("", squashed)))


def _canonical_formula(formula: str) -> str:
    squashed = re.sub(r"(?:\\quad|\s)+", "", formula)
    return re.sub(r"[.,]+(?=\\tag|\\end|\$)", "", squashed).casefold()


def _canonical_cell(cell: str) -> str:
    return re.sub(r"[\s\-]", "", cell).casefold()


def _share(kept: int, expected: Counter[str]) -> float:
    total = sum(expected.values())
    return round(kept / total, 4) if total else 1.0


def combine(parts: list[CleaningMetrics]) -> CleaningMetrics:
    totals = {
        name: sum(getattr(part, name) for part in parts)
        for name, info in CleaningMetrics.model_fields.items()
        if info.annotation is int
    }
    expected = totals["numbers_expected"]
    return CleaningMetrics(
        **totals,
        numbers_preserved_share=round((expected - totals["numbers_lost"]) / max(expected, 1), 4),
        formulas_preserved_share=_weighted(parts, "formulas_preserved_share", "formulas"),
        table_cells_preserved_share=_weighted(parts, "table_cells_preserved_share", "table_cells"),
        noise_share=round(totals["noise_characters"] / max(totals["characters"], 1), 4),
    )


def _weighted(parts: list[CleaningMetrics], share: str, weight: str) -> float:
    total = sum(getattr(part, weight) for part in parts)
    if not total:
        return 1.0
    return round(sum(getattr(part, share) * getattr(part, weight) for part in parts) / total, 4)
