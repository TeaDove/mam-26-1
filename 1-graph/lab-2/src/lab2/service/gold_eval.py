import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import jiwer

from lab2.dto.models import Chunk, GoldPageMetrics
from lab2.service.cleaning import Cleaner, CleaningConfig
from lab2.service.lemmatization import Lemmatizer, english_terms, russian_terms
from lab2.service.normalization import Glossary, Normalizer
from lab2.util.text import count_tokens, language, parse_blocks

KEEP_COVERAGE = 0.6


@dataclass
class GoldPage:
    name: str
    book: str
    raw: str
    gold: str
    units: list[str]
    terms: list[str]


def load_pages(directory: Path) -> list[GoldPage]:
    pages: list[GoldPage] = []
    for gold_path in sorted(directory.glob("*.gold.md")):
        name = gold_path.name.removesuffix(".gold.md")
        units_path = directory / f"{name}.units.txt"
        units = _lines(units_path)
        terms = _lines(directory / f"{name}.terms.txt")
        pages.append(
            GoldPage(
                name=name,
                book=name.split("_p")[0],
                raw=(directory / f"{name}.raw.md").read_text(encoding="utf-8"),
                gold=gold_path.read_text(encoding="utf-8"),
                units=units,
                terms=terms,
            )
        )
    return pages


@dataclass
class GoldEvaluator:
    cleaning: CleaningConfig
    glossary: Glossary

    def evaluate(self, page: GoldPage, book_text: str, clean_book_text: str) -> GoldPageMetrics:
        chunk = Chunk(
            chunk_id=f"{page.book}_gold",
            book=page.book,
            lang=language(book_text),
            section="",
            text=page.raw,
            n_tokens=count_tokens(page.raw),
            start=0,
            end=len(page.raw),
        )
        cleaned = Cleaner(config=self.cleaning).clean([chunk], {page.book: book_text})
        clean_text = cleaned[0].text if cleaned else ""
        lang = language(book_text)
        terms = russian_terms(clean_book_text) if lang == "ru" else english_terms(clean_book_text)
        normalizer = Normalizer(glossary=self.glossary, lemmatizer=Lemmatizer(terms={page.book: terms}))
        normalized = normalizer.normalize(cleaned) if cleaned else []
        normalized_text = _flat(normalized[0].text) if normalized else ""
        should_delete, deleted, correct = self._deletions(page, clean_text)
        units_found = sum(1 for unit in page.units if _flat(unit) in normalized_text)
        terms_found = sum(1 for term in page.terms if _flat(term) in normalized_text)
        precision = correct / deleted if deleted else 1.0
        recall = correct / should_delete if should_delete else 1.0
        return GoldPageMetrics(
            gold_characters=len(page.gold),
            raw_cer=round(jiwer.cer(_flat(page.gold), _flat(page.raw)), 4),
            clean_cer=round(jiwer.cer(_flat(page.gold), _flat(clean_text)), 4),
            raw_wer=round(jiwer.wer(_flat(page.gold), _flat(page.raw)), 4),
            clean_wer=round(jiwer.wer(_flat(page.gold), _flat(clean_text)), 4),
            blocks=len(parse_blocks(page.raw)),
            blocks_should_delete=should_delete,
            blocks_deleted=deleted,
            blocks_deleted_correctly=correct,
            deletion_precision=round(precision, 4),
            deletion_recall=round(recall, 4),
            deletion_f1=round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            units_expected=len(page.units),
            units_correct=units_found,
            unit_accuracy=round(units_found / len(page.units), 4) if page.units else None,
            terms_expected=len(page.terms),
            terms_correct=terms_found,
            term_accuracy=round(terms_found / len(page.terms), 4) if page.terms else None,
        )

    def _deletions(self, page: GoldPage, clean_text: str) -> tuple[int, int, int]:
        should_delete = deleted = correct = 0
        for block in parse_blocks(page.raw):
            must_go = _coverage(block.text, page.gold) < KEEP_COVERAGE
            gone = _coverage(block.text, clean_text) < KEEP_COVERAGE
            should_delete += must_go
            deleted += gone
            correct += must_go and gone
        return should_delete, deleted, correct


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _coverage(fragment: str, text: str) -> float:
    source, target = _flat(fragment), _flat(text)
    if not source:
        return 1.0
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks() if block.size >= 4)
    return matched / len(source)


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
