import re
from collections import Counter
from functools import cache

import tiktoken

from lab2.dto.models import Block, BlockKind

DISPLAY_MATH = re.compile(r"^[ \t]*\$\$[ \t]*\n.*?\n[ \t]*\$\$[ \t]*$", re.S | re.M)
INLINE_MATH = re.compile(r"(?<!\\)\$[^$\n]+?(?<!\\)\$")
CITATION = re.compile(r"\$\s*\^\s*\{\s*[\d,\s\-–]+\}\s*\$")
TABLE = re.compile(r"<table>.*?</table>", re.S)
TABLE_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")
TERMINAL = re.compile(r"[.!?;:…][»”\")\]]*\s*$|\$\$\s*$|</table>\s*$")
CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
CONTINUATION_START = re.compile(r"^(?:[a-zа-яё\d]|\$(?!\$))")
CAPTION = re.compile(r"^(?:#+\s*)?(?:\d{1,2}\s+(?:[A-Z]|\$|[a-z]\s)|Рис\.\s*\d)")
CAPTION_HEADING = re.compile(r"^#+\s*\d{1,2}\s+[A-Z]")
PANEL_LIST = re.compile(r"^(?:\$?[a-d]\$?\s.*?[;,]\s*\$?b\$?\s|\$[a-d],\s?[a-d])", re.S)
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
DANGLING_WORD = re.compile(
    r"\b(?:in|of|the|a|an|and|or|to|by|with|from|for|as|at|on|is|are|was|were|that|than"
    r"|в|на|и|с|по|к|от|до|для|при|что|как)$",
    re.I,
)
ABBREVIATION_END = re.compile(
    r"(?:\b(?:Fig|Figs|Eq|Eqs|Ref|Refs|al|vs|v|No|Nos|pp|approx|рис|табл|см|др|им|т\.к|т\.е|т\.н|г|гг|с|ст)"
    r"|\b[A-ZА-ЯЁ])\.\s*$"
)


@cache
def encoder(name: str = "o200k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, name: str = "o200k_base") -> int:
    return len(encoder(name).encode(text, disallowed_special=()))


def language(text: str) -> str:
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return "en"
    cyrillic = sum(1 for letter in letters if CYRILLIC.match(letter))
    return "ru" if cyrillic / len(letters) > 0.3 else "en"


def parse_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    start: int | None = None
    end = 0
    in_formula = False
    offset = 0
    for line in markdown.split("\n"):
        line_start, line_end = offset, offset + len(line)
        offset = line_end + 1
        stripped = line.strip()
        if in_formula:
            end = line_end
            if stripped == "$$":
                blocks.append(_block(markdown, start or 0, end))
                start, in_formula = None, False
            continue
        if stripped == "$$":
            _flush(markdown, start, end, blocks)
            start, end, in_formula = line_start, line_end, True
            continue
        if not stripped:
            _flush(markdown, start, end, blocks)
            start = None
            continue
        if start is None:
            start = line_start
        end = line_end
    _flush(markdown, start, end, blocks)
    return blocks


def _flush(markdown: str, start: int | None, end: int, blocks: list[Block]) -> None:
    if start is not None:
        blocks.append(_block(markdown, start, end))


def _block(markdown: str, start: int, end: int) -> Block:
    text = markdown[start:end].strip()
    return Block(kind=block_kind(text), text=text, start=start, end=end)


def block_kind(text: str) -> BlockKind:
    if text.startswith("#") and not CAPTION_HEADING.match(text):
        return BlockKind.HEADING
    if text.startswith("$$"):
        return BlockKind.FORMULA
    if text.startswith("<table"):
        return BlockKind.TABLE
    if text.startswith("!["):
        return BlockKind.IMAGE
    return BlockKind.PARAGRAPH


def furniture_lines(markdown: str, min_repeats: int = 3, max_chars: int = 80) -> frozenset[str]:
    lines = Counter(line.strip() for line in markdown.split("\n") if line.strip())
    return frozenset(
        line
        for line, repeats in lines.items()
        if repeats >= min_repeats and len(line) <= max_chars and not line.startswith("$$")
    )


def is_furniture(block: Block, furniture: frozenset[str]) -> bool:
    text = block.text.strip()
    return text in furniture or bool(PAGE_NUMBER.match(text)) or text.startswith("<small>")


def is_caption(block: Block) -> bool:
    return block.kind is BlockKind.PARAGRAPH and bool(CAPTION.match(block.text) or PANEL_LIST.match(block.text))


def is_body(block: Block, furniture: frozenset[str]) -> bool:
    if block.kind in (BlockKind.FORMULA, BlockKind.TABLE):
        return True
    return block.kind is BlockKind.PARAGRAPH and not is_furniture(block, furniture) and not is_caption(block)


def ends_sentence(text: str) -> bool:
    return bool(TERMINAL.search(text.rstrip()))


def ends_sentence_strict(text: str) -> bool:
    stripped = text.rstrip()
    return ends_sentence(stripped) and not ABBREVIATION_END.search(stripped)


def starts_mid_sentence(text: str) -> bool:
    return bool(CONTINUATION_START.match(text.lstrip()))


def is_continuation(left: str, right: str) -> bool:
    stripped = left.rstrip()
    if ends_sentence_strict(stripped):
        return False
    return starts_mid_sentence(right) or stripped.endswith("-") or bool(DANGLING_WORD.search(stripped))


def clean_cut(left: list[Block], right: list[Block], furniture: frozenset[str] = frozenset()) -> bool:
    left_blocks = [b for b in left if is_body(b, furniture) or b.kind is BlockKind.IMAGE]
    right_blocks = [b for b in right if is_body(b, furniture) or b.kind is BlockKind.HEADING]
    if not left_blocks or not right_blocks:
        return False
    last, first = left_blocks[-1], right_blocks[0]
    if first.kind is BlockKind.HEADING:
        return last.kind is not BlockKind.IMAGE and (last.kind is not BlockKind.PARAGRAPH or ends_sentence(last.text))
    if last.kind is BlockKind.IMAGE or first.kind is BlockKind.FORMULA:
        return False
    if last.kind in (BlockKind.FORMULA, BlockKind.TABLE):
        return not starts_mid_sentence(first.text)
    return not is_continuation(last.text, first.text)


def formulas(text: str) -> list[str]:
    display = DISPLAY_MATH.findall(text)
    inline = INLINE_MATH.findall(DISPLAY_MATH.sub(" ", text))
    return display + [f for f in inline if not is_citation(f)]


def table_cells(text: str) -> list[str]:
    cells: list[str] = []
    for table in TABLE.findall(text):
        cells.extend(cell.strip() for cell in TABLE_CELL.findall(table) if cell.strip())
    return cells


def is_citation(formula: str) -> bool:
    return bool(CITATION.fullmatch(formula.strip()))
