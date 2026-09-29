from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lab2.dto.models import Chunk
from lab2.util.text import count_tokens, language


@dataclass
class Corpus:
    raw_dir: Path
    books: tuple[str, ...]

    def raw_text(self, book: str) -> str:
        return (self.raw_dir / f"{book}.md").read_text(encoding="utf-8")

    def dirty_chunks(self, graphrag_output: Path) -> list[Chunk]:
        units = pd.read_parquet(graphrag_output / "text_units.parquet")
        documents = pd.read_parquet(graphrag_output / "documents.parquet")
        titles = dict(zip(documents["id"], documents["title"], strict=True))
        units["book"] = units["document_id"].map(lambda doc: Path(titles[doc]).stem)
        chunks: list[Chunk] = []
        for book in self.books:
            source = self.raw_text(book)
            cursor = 0
            rows = units[units["book"] == book].sort_values("human_readable_id")
            for index, row in enumerate(rows.itertuples()):
                text = str(row.text)
                start = source.find(text.strip()[:120], max(cursor - len(text), 0))
                if start < 0:
                    raise ValueError(f"text unit {row.human_readable_id} not found in {book}")
                cursor = start + 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{book}_u{index:03d}",
                        book=book,
                        lang=language(source),
                        section="",
                        text=text,
                        n_tokens=count_tokens(text),
                        start=start,
                        end=start + len(text.strip()),
                    )
                )
        return chunks
