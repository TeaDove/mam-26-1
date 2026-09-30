from pathlib import Path

import pytest

from lab2.dto.models import Chunk
from lab2.service.chunking import StructuralChunker
from lab2.service.cleaning import BookRules, Cleaner, CleaningConfig
from lab2.service.normalization import Glossary, Normalizer
from lab2.service.vectorization import word_windows

ROOT = Path(__file__).resolve().parents[1]


def _chunk(text: str, book: str = "tanaka1981", lang: str = "en") -> Chunk:
    return Chunk(chunk_id=f"{book}_c000", book=book, lang=lang, section="", text=text, n_tokens=0, start=0, end=0)


@pytest.fixture
def cleaner() -> Cleaner:
    rules = BookRules(
        drop_blocks=[], drop_sections=["REFERENCES"], references_start=None, bibliography_start="^## REFERENCES"
    )
    config = CleaningConfig(
        running_header_min_repeats=3,
        running_header_max_chars=80,
        dehyphenation_keep_tails=["and", "и"],
        books={"tanaka1981": rules, "stat3": rules},
        ocr_fixes=[],
    )
    return Cleaner(config=config)


def test_chunker_loses_no_text() -> None:
    paragraphs = [f"Paragraph number {i} has several words and ends properly." for i in range(200)]
    text = "# Title\n\n" + "\n\n".join(paragraphs)
    chunks = StructuralChunker(target_tokens=200, min_tokens=50, hard_max_tokens=260).chunk("tanaka1981", text)
    assert len(chunks) > 1
    assert "\n\n".join(c.text for c in chunks) == text
    assert all(c.n_tokens <= 260 for c in chunks)


def test_cleaner_removes_furniture_and_citations(cleaner: Cleaner) -> None:
    raw = "\n\n".join(["Header line"] * 3 + ["185", "Text by Arrowsmith $^{1}$ on grains [2]."])
    chunk = _chunk(raw)
    cleaned = cleaner.clean([chunk], {"tanaka1981": raw})
    assert cleaned[0].text == "Text by Arrowsmith on grains."


def test_cleaner_merges_broken_paragraph_around_caption(cleaner: Cleaner) -> None:
    raw = "Steels were rolled at con-\n\n12 Effect of temperature on grain size\n\ntrolled temperatures."
    cleaned = cleaner.clean([_chunk(raw)], {"tanaka1981": raw})
    assert cleaned[0].text.startswith("Steels were rolled at controlled temperatures.")
    assert "12 Effect of temperature" in cleaned[0].text


def test_cleaner_keeps_abbreviation_hyphen_in_table(cleaner: Cleaner) -> None:
    raw = "<table><tr><td>Т-ра на-грева</td></tr></table>"
    cleaned = cleaner.clean([_chunk(raw, "stat3", "ru")], {"stat3": raw})
    assert "Т-ра нагрева" in cleaned[0].text


@pytest.fixture
def normalizer() -> Normalizer:
    return Normalizer(glossary=Glossary.load(ROOT / "configs" / "glossary.yaml"))


def test_temperature_math_becomes_plain_text(normalizer: Normalizer) -> None:
    result = normalizer.normalize([_chunk("нагрев до $850 ^ { 0 } \\mathbf { C } ,$ затем", "stat3", "ru")])
    assert "850 °C," in result[0].text


def test_canonical_english_goes_first_once(normalizer: Normalizer) -> None:
    text = "Режим КП и снова КП на стане."
    result = normalizer.normalize([_chunk(text, "stat3", "ru")])[0].text
    assert result.startswith("Режим controlled rolling (КП, контролируемая прокатка) и снова КП")


def test_formulas_and_element_formulas_are_protected(normalizer: Normalizer) -> None:
    text = "Precipitation of Nb(C, N) in Mn-Nb-V steel.\n\n$$\n44(\\%\\mathrm{Si})\n$$"
    result = normalizer.normalize([_chunk(text)])[0].text
    assert "Nb(C, N)" in result
    assert "Mn–Nb–V" in result
    assert "\\mathrm{Si})" in result


def test_word_windows() -> None:
    assert word_windows(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]
