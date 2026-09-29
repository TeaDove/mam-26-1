import re
from dataclasses import dataclass, field

from lab2.dto.models import Block, BlockKind, Chunk
from lab2.util.text import clean_cut, count_tokens, furniture_lines, language, parse_blocks

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ(])")


@dataclass
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class StructuralChunker:
    target_tokens: int
    min_tokens: int
    hard_max_tokens: int
    furniture: frozenset[str] = frozenset()

    def chunk(self, book: str, markdown: str) -> list[Chunk]:
        lang = language(markdown)
        self.furniture = furniture_lines(markdown)
        groups: list[tuple[str, list[Block]]] = []
        for section in self._merge_small(self._sections(parse_blocks(markdown))):
            groups.extend((section.title, piece) for piece in self._pack(self._split_oversized(section.blocks)))
        groups = self._merge_tiny(groups)
        return [
            Chunk(
                chunk_id=f"{book}_c{index:03d}",
                book=book,
                lang=lang,
                section=title,
                text=_join(blocks),
                n_tokens=count_tokens(_join(blocks)),
                start=blocks[0].start,
                end=blocks[-1].end,
            )
            for index, (title, blocks) in enumerate(groups)
        ]

    def _sections(self, blocks: list[Block]) -> list[Section]:
        sections: list[Section] = [Section(title="")]
        for block in blocks:
            current = sections[-1]
            starts_new = block.kind is BlockKind.HEADING and any(
                b.kind is not BlockKind.HEADING for b in current.blocks
            )
            if starts_new:
                sections.append(Section(title=block.text.lstrip("# ").strip()))
            elif block.kind is BlockKind.HEADING and not current.title:
                current.title = block.text.lstrip("# ").strip()
            sections[-1].blocks.append(block)
        return [section for section in sections if section.blocks]

    def _merge_small(self, sections: list[Section]) -> list[Section]:
        merged: list[Section] = []
        for section in sections:
            previous = merged[-1] if merged else None
            if (
                previous
                and self._tokens(previous.blocks) < self.min_tokens
                and self._tokens(previous.blocks + section.blocks) <= self.target_tokens
            ):
                previous.blocks.extend(section.blocks)
                continue
            merged.append(section)
        return merged

    def _pack(self, blocks: list[Block]) -> list[list[Block]]:
        pieces: list[list[Block]] = []
        current: list[Block] = []
        for block in blocks:
            if current and self._tokens([*current, block]) > self.target_tokens:
                cut = self._best_cut(current, block)
                pieces.append(current[:cut])
                current = current[cut:]
            current.append(block)
        if current:
            pieces.append(current)
        if (
            len(pieces) > 1
            and self._tokens(pieces[-1]) < self.min_tokens
            and self._tokens(pieces[-2] + pieces[-1]) <= self.hard_max_tokens
        ):
            pieces[-2].extend(pieces.pop())
        return pieces

    def _merge_tiny(self, groups: list[tuple[str, list[Block]]]) -> list[tuple[str, list[Block]]]:
        merged: list[tuple[str, list[Block]]] = []
        pending: list[Block] = []
        for title, blocks in groups:
            blocks = pending + blocks
            pending = []
            if self._tokens(blocks) >= self.min_tokens:
                merged.append((title, blocks))
                continue
            if merged and self._tokens(merged[-1][1] + blocks) <= self.hard_max_tokens:
                merged[-1] = (merged[-1][0], merged[-1][1] + blocks)
            else:
                pending = blocks
        if pending:
            merged.append(("", pending))
        return merged

    def _best_cut(self, current: list[Block], incoming: Block) -> int:
        candidates = [*current, incoming]
        for cut in range(len(current), 0, -1):
            if self._tokens(current[:cut]) < self.min_tokens:
                break
            if clean_cut(candidates[:cut], candidates[cut:], self.furniture):
                return cut
        return len(current)

    def _split_oversized(self, blocks: list[Block]) -> list[Block]:
        result: list[Block] = []
        for block in blocks:
            if block.kind is BlockKind.PARAGRAPH and self._tokens([block]) > self.hard_max_tokens:
                result.extend(_split_sentences(block, self.target_tokens))
            else:
                result.append(block)
        return result

    def _tokens(self, blocks: list[Block]) -> int:
        return count_tokens(_join(blocks))


def _split_sentences(block: Block, target_tokens: int) -> list[Block]:
    parts: list[Block] = []
    buffer: list[str] = []
    for sentence in SENTENCE_SPLIT.split(block.text):
        if buffer and count_tokens(" ".join([*buffer, sentence])) > target_tokens:
            parts.append(_sub_block(block, " ".join(buffer)))
            buffer = []
        buffer.append(sentence)
    if buffer:
        parts.append(_sub_block(block, " ".join(buffer)))
    return parts


def _sub_block(block: Block, text: str) -> Block:
    offset = block.start + max(block.text.find(text[:40]), 0)
    return Block(kind=block.kind, text=text, start=offset, end=offset + len(text))


def _join(blocks: list[Block]) -> str:
    return "\n\n".join(block.text for block in blocks)
