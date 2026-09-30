import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel
from wordfreq import zipf_frequency

from lab2.dto.models import AuditEntry, Block, BlockKind, Chunk
from lab2.util.text import (
    CITATION,
    INLINE_MATH,
    PAGE_NUMBER,
    count_tokens,
    furniture_lines,
    is_caption,
    is_continuation,
    parse_blocks,
)

CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏﻿]")
BRACKET_CITATION = re.compile(r"\s*\[\d+(?:\s*[,–\-]\s*\d+)*\]")
DOLLAR_CITATION = re.compile(r"\s*" + CITATION.pattern)
MIXED_CITATION = re.compile(r",\s*\^\s*\{\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\}\s*(?=\$)")
GLUED_CITATION = re.compile(r"(?<=et al\.)\d{1,3}\b")
JOURNAL_CODE = re.compile(r"\s*\bIMR/\d+\s*$")
INLINE_HYPHEN = re.compile(r"([^\W\d_]+)-[ \t]+([^\W\d_]+)")
CELL_HYPHEN = re.compile(r"([а-яё]{2,})-([а-яё]+)", re.I)
MARKUP_TAG = re.compile(r"</?(?:strong|u|b|i|em)>")
ESCAPE = re.compile(r"\\([*~_])")
LEADER_DOTS = re.compile(r"(?:\s*(?:\\quad\s*)?\.){2,}\s*,?")
SPACED_DIGITS = re.compile(r"(?<=\d) (?=\d)")
SPACED_DECIMAL = re.compile(r"(\d)\. (\d)")
MULTISPACE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT = re.compile(r"(?<=[^\s$]) +([,;:.])(?=\s|$)")
PAGE_FOOTNOTE = re.compile(r"^<small>.*docvortex-page-footnote", re.S)
DELETION_RULES = frozenset(
    {
        "image_placeholder",
        "page_footnote",
        "running_header",
        "page_number",
        "metadata_block",
        "dropped_section",
        "reference_list",
        "citation_marker",
        "journal_code",
    }
)
MAX_CAPTIONS_BETWEEN = 5


class OcrFix(BaseModel):
    source: str
    target: str


class BookRules(BaseModel):
    drop_blocks: list[str]
    drop_sections: list[str]
    references_start: str | None
    bibliography_start: str


class CleaningConfig(BaseModel):
    running_header_min_repeats: int
    running_header_max_chars: int
    dehyphenation_keep_tails: list[str]
    books: dict[str, BookRules]
    ocr_fixes: list[OcrFix]

    @classmethod
    def load(cls, path: Path) -> CleaningConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class BookState:
    rules: BookRules
    headers: frozenset[str]
    vocabulary: Counter[str]
    compounds: frozenset[str]
    dropping_section: bool = False
    in_references: bool = False


@dataclass
class Cleaner:
    config: CleaningConfig
    audit: list[AuditEntry] = field(default_factory=list)

    def clean(self, chunks: list[Chunk], raw_books: dict[str, str]) -> list[Chunk]:
        states = {book: self._state(book, text) for book, text in raw_books.items()}
        cleaned: list[Chunk] = []
        for chunk in chunks:
            text = self._clean_chunk(chunk, states[chunk.book])
            if not text.strip():
                continue
            cleaned.append(chunk.model_copy(update={"text": text, "n_tokens": count_tokens(text)}))
        return cleaned

    def headers_for(self, raw: str) -> frozenset[str]:
        return furniture_lines(raw, self.config.running_header_min_repeats, self.config.running_header_max_chars)

    def _state(self, book: str, raw: str) -> BookState:
        vocabulary = Counter(word.casefold() for word in re.findall(r"[^\W\d_]+", raw))
        compounds = frozenset(word.casefold() for word in re.findall(r"[^\W\d_]+-[^\W\d_]+(?:-[^\W\d_]+)*", raw))
        return BookState(
            rules=self.config.books[book],
            headers=self.headers_for(raw),
            vocabulary=vocabulary,
            compounds=compounds,
        )

    def _clean_chunk(self, chunk: Chunk, state: BookState) -> str:
        kept: list[Block] = []
        for block in parse_blocks(chunk.text):
            rule = self._deletion_rule(block, state)
            if rule:
                self._log(chunk.chunk_id, rule, block.text, "")
                continue
            kept.append(block)
        texts = [
            self._clean_inline(chunk.chunk_id, block, state)
            for block in self._merge_broken(chunk.chunk_id, kept, state)
        ]
        return "\n\n".join(text for text in texts if text.strip())

    def _deletion_rule(self, block: Block, state: BookState) -> str | None:
        text = block.text.strip()
        if block.kind is BlockKind.HEADING:
            title = text.lstrip("# ").strip()
            state.dropping_section = title in state.rules.drop_sections
        if state.rules.references_start and re.match(state.rules.references_start, text):
            state.in_references = True
        checks: list[tuple[str, bool]] = [
            ("reference_list", state.in_references),
            ("dropped_section", state.dropping_section),
            ("image_placeholder", block.kind is BlockKind.IMAGE),
            ("page_footnote", bool(PAGE_FOOTNOTE.match(text))),
            ("running_header", text in state.headers),
            ("page_number", bool(PAGE_NUMBER.match(text))),
            ("metadata_block", any(re.match(pattern, text) for pattern in state.rules.drop_blocks)),
        ]
        return next((rule for rule, hit in checks if hit), None)

    def _merge_broken(self, chunk_id: str, blocks: list[Block], state: BookState) -> list[Block]:
        result: list[Block] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]
            follower, captions = None, []
            if block.kind is BlockKind.PARAGRAPH and not is_caption(block):
                follower, captions = self._continuation(block, blocks, index + 1)
            if follower is None:
                result.append(block)
                index += 1
                continue
            joined = self._join_broken(block.text, blocks[follower].text, state)
            self._log(
                chunk_id,
                "broken_paragraph",
                block.text[-80:] + " ¶ " + blocks[follower].text[:80],
                joined[:200],
            )
            merged = Block(kind=BlockKind.PARAGRAPH, text=joined, start=block.start, end=blocks[follower].end)
            blocks = [*blocks[:index], merged, *captions, *blocks[follower + 1 :]]
        return result

    def _continuation(self, left: Block, blocks: list[Block], start: int) -> tuple[int | None, list[Block]]:
        captions: list[Block] = []
        for index in range(start, min(start + MAX_CAPTIONS_BETWEEN + 1, len(blocks))):
            candidate = blocks[index]
            if is_caption(candidate):
                captions.append(candidate)
                continue
            if candidate.kind is BlockKind.PARAGRAPH and is_continuation(left.text, candidate.text):
                return index, captions
            return None, []
        return None, []

    def _join_broken(self, left: str, right: str, state: BookState) -> str:
        head = left.rstrip()
        stem = re.search(r"([^\W\d_]+)-$", head)
        tail = re.match(r"[^\W\d_]+", right.lstrip())
        if stem and tail:
            return (
                head[: -len(stem.group(0))]
                + self._rejoin(stem.group(1), tail.group(0), state)
                + right.lstrip()[len(tail.group(0)) :]
            )
        return head + " " + right.lstrip()

    def _rejoin(self, stem: str, tail: str, state: BookState) -> str:
        compound = f"{stem}-{tail}".casefold()
        if tail.casefold() in self.config.dehyphenation_keep_tails:
            return f"{stem}- {tail}"
        if compound in state.compounds:
            return f"{stem}-{tail}"
        if re.search(r"[а-яё]", stem + tail, re.I):
            return f"{stem}-{tail}" if tail[0].isupper() else stem + tail
        joined = (stem + tail).casefold()
        if state.vocabulary[joined] or zipf_frequency(joined, "en") > 0:
            return stem + tail
        return f"{stem}-{tail}"

    def _clean_inline(self, chunk_id: str, block: Block, state: BookState) -> str:
        text = block.text
        text = self._apply(chunk_id, "unicode_nfc", text, lambda t: unicodedata.normalize("NFC", t))
        text = self._apply(chunk_id, "control_characters", text, lambda t: CONTROL.sub("", t))
        for fix in self.config.ocr_fixes:
            replacement = fix.target.replace("\\", "\\\\")
            text = self._sub_each(chunk_id, "ocr_fix", re.compile(re.escape(fix.source)), replacement, text)
        text = self._sub_each(chunk_id, "citation_marker", DOLLAR_CITATION, "", text)
        text = self._sub_each(chunk_id, "citation_marker", BRACKET_CITATION, "", text)
        text = self._sub_each(chunk_id, "citation_marker", MIXED_CITATION, "", text)
        text = self._sub_each(chunk_id, "citation_marker", GLUED_CITATION, "", text)
        text = self._sub_each(chunk_id, "journal_code", JOURNAL_CODE, "", text)
        text = self._sub_each(chunk_id, "markup_tag", MARKUP_TAG, "", text)
        text = self._dehyphenate(chunk_id, text, state)
        if block.kind is BlockKind.TABLE:
            text = self._dehyphenate_cells(chunk_id, text, state)
        if block.kind is not BlockKind.FORMULA:
            text = self._outside_math(text, lambda t: self._sub_each(chunk_id, "markdown_escape", ESCAPE, r"\1", t))
        text = self._clean_math(chunk_id, text)
        if block.kind is not BlockKind.FORMULA:
            text = self._sub_each(chunk_id, "whitespace", MULTISPACE, " ", text)
            text = self._sub_each(chunk_id, "space_before_punctuation", SPACE_BEFORE_PUNCT, r"\1", text)
        return text.strip()

    def _dehyphenate(self, chunk_id: str, text: str, state: BookState) -> str:
        def replace(match: re.Match[str]) -> str:
            after = self._rejoin(match.group(1), match.group(2), state)
            if after == match.group(0):
                return after
            self._log(chunk_id, "line_break_hyphen", match.group(0), after)
            return after

        return INLINE_HYPHEN.sub(replace, text)

    def _dehyphenate_cells(self, chunk_id: str, text: str, state: BookState) -> str:
        def replace(match: re.Match[str]) -> str:
            joined = match.group(1) + match.group(2)
            known = state.vocabulary[joined.casefold()] or zipf_frequency(joined.casefold(), "ru") >= 1.5
            if not known:
                return match.group(0)
            self._log(chunk_id, "table_cell_hyphen", match.group(0), joined)
            return joined

        return CELL_HYPHEN.sub(replace, text)

    def _clean_math(self, chunk_id: str, text: str) -> str:
        def fix(formula: str) -> str:
            result = LEADER_DOTS.sub(" ", formula)
            result = SPACED_DECIMAL.sub(r"\1.\2", result)
            previous = None
            while previous != result:
                previous, result = result, SPACED_DIGITS.sub("", result)
            if result != formula:
                self._log(chunk_id, "formula_noise", formula[:200], result[:200])
            return result

        if text.startswith("$$"):
            return fix(text)
        return INLINE_MATH.sub(lambda m: fix(m.group(0)), text)

    def _outside_math(self, text: str, function: Callable[[str], str]) -> str:
        pieces: list[str] = []
        cursor = 0
        for match in INLINE_MATH.finditer(text):
            pieces.extend([function(text[cursor : match.start()]), match.group(0)])
            cursor = match.end()
        pieces.append(function(text[cursor:]))
        return "".join(pieces)

    def _sub_each(self, chunk_id: str, rule: str, pattern: re.Pattern[str], replacement: str, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            after = match.expand(replacement)
            self._log(chunk_id, rule, match.group(0), after)
            return after

        return pattern.sub(replace, text)

    def _apply(self, chunk_id: str, rule: str, text: str, function: Callable[[str], str]) -> str:
        result = function(text)
        if result != text:
            self._log(chunk_id, rule, text[:200], result[:200])
        return result

    def _log(self, chunk_id: str, rule: str, before: str, after: str) -> None:
        self.audit.append(AuditEntry(chunk_id=chunk_id, rule=rule, before=before, after=after))
