from itertools import pairwise
from pathlib import Path

from lab2.dto.models import Chunk
from lab2.service.corpus import strip_bibliography, token_windows
from lab2.service.lemmatization import Lemmatizer
from lab2.service.normalization import Glossary, Normalizer
from lab2.service.retrieval import Query, RankedQuery

ROOT = Path(__file__).resolve().parents[1]


def _chunk(text: str, lang: str) -> Chunk:
    return Chunk(chunk_id="stat3_all", book="stat3", lang=lang, section="", text=text, n_tokens=0, start=0, end=0)


def test_strip_bibliography_cuts_from_heading() -> None:
    text = "Body text.\n\n## REFERENCES\n\n1. A. Author: Journal."
    assert strip_bibliography(text, "^## REFERENCES") == "Body text.\n"


def test_token_windows_cover_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(3000))
    windows = token_windows("tanaka1981", text, 500, 50)
    assert windows[0].start == 0
    assert windows[-1].end == len(text)
    assert all(text[w.start : w.end] == w.text for w in windows)
    assert all(later.start < earlier.end for earlier, later in pairwise(windows))


def test_russian_lemmatization_agrees_adjective_with_noun() -> None:
    lemmatizer = Lemmatizer()
    result = lemmatizer.lemmatize("stat3_all", "прокатка в чистовой клети стана", "ru")
    assert result == "прокатка в чистовая клеть стан"


def test_english_lemmatization_only_touches_plural_nouns() -> None:
    lemmatizer = Lemmatizer()
    result = lemmatizer.lemmatize("tanaka1981_all", "The grains are coarse and separations occur.", "en")
    assert result == "The grain are coarse and separation occur."


def test_canonical_term_is_repeated_after_distance() -> None:
    normalizer = Normalizer(glossary=Glossary.load(ROOT / "configs" / "glossary.yaml"), repeat_chars=60)
    text = "аустенит растёт. " + "x " * 40 + "аустенит снова. аустенит рядом."
    result = normalizer.normalize([_chunk(text, "ru")])[0].text
    assert result.count("austenite (") == 2


def test_ndcg_and_first_hit() -> None:
    query = Query(id="q", lang="en", book="tanaka1981", question="?", evidence=["x"], source="test")
    ranked = RankedQuery(query=query, ranking=[3, 1, 2], relevant={1})
    assert ranked.first_hit() == 2
    assert round(ranked.ndcg(10), 4) == round(1 / 1.5849625007211562, 4)
