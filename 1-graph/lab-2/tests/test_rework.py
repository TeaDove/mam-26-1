from itertools import pairwise
from pathlib import Path

import pandas as pd

from lab2.api.stages import _source_fragment
from lab2.dto.models import Chunk
from lab2.service.corpus import strip_bibliography, token_windows
from lab2.service.graph_criteria import noise_labels, source_vocabulary
from lab2.service.graph_eval import load_graph
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
    lemmatizer = Lemmatizer(terms={"stat3": frozenset({"клеть", "стан", "прокатка", "сталь"})})
    result = lemmatizer.lemmatize("stat3_all", "прокатка в черновых клетях стана, в том числе стали", "ru")
    assert result == "прокатка в черновая клеть стан, в том числе сталь"


def test_english_lemmatization_only_touches_plural_nouns() -> None:
    lemmatizer = Lemmatizer(terms={"tanaka1981": frozenset({"grain", "separation"})})
    result = lemmatizer.lemmatize("tanaka1981_all", "Nb retards grains and separations occur in data.", "en")
    assert result == "Nb retards grain and separation occur in data."


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


def test_steel_composition_stays_one_term() -> None:
    normalizer = Normalizer(glossary=Glossary.load(ROOT / "configs" / "glossary.yaml"))
    text = "The C—Mn—Nb fine-grained steel and 0.06C-0.2Si-1.96Mn steel differ from Mn steel."
    chunk = Chunk(
        chunk_id="tanaka1981_all", book="tanaka1981", lang="en", section="", text=text, n_tokens=0, start=0, end=0
    )
    result = normalizer.normalize([chunk])[0].text
    assert "C–Mn–Nb fine-grained steel" in result
    assert "0.06C–0.2Si–1.96Mn steel" in result
    assert "Mn (manganese) steel" in result


def test_reverse_relationships_keep_both_descriptions(tmp_path: Path) -> None:
    pd.DataFrame({"id": ["d"], "title": ["tanaka1981_c001.md"]}).to_parquet(tmp_path / "documents.parquet")
    pd.DataFrame({"id": ["u"], "document_id": ["d"], "text": ["x"], "n_tokens": [1]}).to_parquet(
        tmp_path / "text_units.parquet"
    )
    pd.DataFrame(
        {"title": ["A", "B"], "type": ["concept"] * 2, "description": ["a", "b"], "frequency": [1, 1]}
        | {"text_unit_ids": [["u"], ["u"]]}
    ).to_parquet(tmp_path / "entities.parquet")
    pd.DataFrame(
        {"source": ["A", "B"], "target": ["B", "A"], "weight": [1.0, 2.0], "description": ["A to B", "B to A"]}
    ).to_parquet(tmp_path / "relationships.parquet")
    graph = load_graph(tmp_path).graph
    edge = graph["A"]["B"]
    assert graph.number_of_edges() == 1
    assert edge["weight"] == 3.0
    assert edge["description"] == "A to B\nB to A"
    assert edge["merged"] == 2


def test_hyphenated_terms_are_not_broken_words() -> None:
    vocabulary = frozenset({"нагрева", "слябов", "температура", "steel"})
    assert noise_labels("C-MN-NB STEEL", vocabulary, frozenset()) == []
    assert noise_labels("HIGH-STRENGTH LOW-ALLOY STEEL", vocabulary, frozenset()) == []
    assert "broken_word" in noise_labels("ТЕМПЕРАТУРА НА-ГРЕВА СЛЯ-БОВ", vocabulary, frozenset())


def test_inflected_source_words_are_known() -> None:
    vocabulary = source_vocabulary(["малоперлитных сталей"])
    assert noise_labels("МАЛОПЕРЛИТНЫЕ СТАЛИ", vocabulary, frozenset()) == []
    assert noise_labels("МАЛОПЕРЛИТНЫЕ СТАЛИЩИ", vocabulary, frozenset()) == ["unknown_word"]


def test_source_fragment_prefers_unit_mentioning_node() -> None:
    texts = {"a": "about austenite", "b": "ferrite and more ferrite", "c": "ferrite"}
    assert _source_fragment("FERRITE", ["a", "c", "b"], texts) == "ferrite and more ferrite"
    assert _source_fragment("PEARLITE", ["a", "b"], texts) == "about austenite"
    assert _source_fragment("PEARLITE", [], texts) == ""
