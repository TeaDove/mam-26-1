import re
from dataclasses import dataclass
from pathlib import Path

from lab2.dto.models import Chunk
from lab2.util.text import encoder, language


@dataclass
class Corpus:
    raw_dir: Path
    books: tuple[str, ...]

    def raw_text(self, book: str) -> str:
        return (self.raw_dir / f"{book}.md").read_text(encoding="utf-8")


def strip_bibliography(text: str, start_pattern: str) -> str:
    match = re.search(start_pattern, text, re.M)
    if not match:
        raise ValueError(f"bibliography start not found: {start_pattern}")
    return text[: match.start()].rstrip() + "\n"


def token_windows(book: str, text: str, size: int, overlap: int) -> list[Chunk]:
    encoding = encoder()
    tokens = encoding.encode(text, disallowed_special=())
    decoded, offsets = encoding.decode_with_offsets(tokens)
    chunks: list[Chunk] = []
    lang = language(text)
    step = size - overlap
    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + size]
        prefix = offsets[start]
        stop = offsets[start + size] if start + size < len(tokens) else len(decoded)
        body = decoded[prefix:stop]
        chunks.append(
            Chunk(
                chunk_id=f"{book}_u{index:03d}",
                book=book,
                lang=lang,
                section="",
                text=body,
                n_tokens=len(window),
                start=prefix,
                end=prefix + len(body),
            )
        )
        if start + size >= len(tokens):
            break
    return chunks
