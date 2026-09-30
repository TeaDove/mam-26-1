import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cache

import pymorphy3
import spacy
from spacy.language import Language

from lab2.dto.models import AuditEntry

CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё]+")
LATIN_WORD = re.compile(r"[A-Za-z]+")
NAME_TAGS = frozenset({"Name", "Surn", "Patr", "Geox", "Orgn"})
NON_TERM_NOUNS = frozenset(
    {
        "число",
        "образ",
        "точка",
        "зрение",
        "случай",
        "время",
        "раз",
        "результат",
        "вид",
        "часть",
        "ряд",
        "сторона",
        "основа",
        "счёт",
        "итог",
        "мера",
        "связь",
        "условие",
        "зависимость",
        "помощь",
        "качество",
        "возможность",
        "значение",
        "уровень",
        "работа",
        "год",
        "рис",
        "таблица",
        "пример",
        "данные",
        "показатель",
        "сравнение",
    }
)
MIN_TERM_FREQUENCY = 3


@cache
def morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@cache
def english() -> Language:
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


@cache
def best_parse(word: str) -> pymorphy3.analyzer.Parse:
    parses = morph().parse(word)
    return next((p for p in parses if not p.tag.grammemes & NAME_TAGS), parses[0])


def russian_terms(text: str) -> frozenset[str]:
    counts: Counter[str] = Counter()
    for word in CYRILLIC_WORD.findall(text.lower()):
        parsed = best_parse(word)
        if parsed.tag.POS == "NOUN":
            counts[parsed.normal_form] += 1
    return frozenset(t for t, c in counts.items() if c >= MIN_TERM_FREQUENCY and len(t) >= 4) - NON_TERM_NOUNS


def english_terms(text: str) -> frozenset[str]:
    counts: Counter[str] = Counter()
    for paragraph in text.split("\n\n"):
        for token in english()(paragraph):
            if token.pos_ == "NOUN" and token.is_alpha and token.tag_ == "NN":
                counts[token.text.lower()] += 1
    return frozenset(t for t, c in counts.items() if c >= MIN_TERM_FREQUENCY and len(t) >= 4)


@dataclass
class LemmaStats:
    words: int = 0
    changed: int = 0
    full_changed: int = 0


@dataclass
class Lemmatizer:
    terms: dict[str, frozenset[str]] = field(default_factory=dict)
    audit: list[AuditEntry] = field(default_factory=list)
    stats: dict[str, LemmaStats] = field(default_factory=dict)

    def lemmatize(self, chunk_id: str, text: str, lang: str) -> str:
        stats = self.stats.setdefault(chunk_id, LemmaStats())
        terms = self.terms.get(chunk_id.rsplit("_", 1)[0], frozenset())
        if lang == "ru":
            return self._russian(chunk_id, text, stats, terms or russian_terms(text))
        return self._english(chunk_id, text, stats, terms or english_terms(text))

    def _russian(self, chunk_id: str, text: str, stats: LemmaStats, terms: frozenset[str]) -> str:
        matches = list(CYRILLIC_WORD.finditer(text))
        parses = [_term_parse(m.group(0), terms) for m in matches]
        replacements: dict[int, str] = {}
        for index, (match, parsed) in enumerate(zip(matches, parses, strict=True)):
            stats.words += 1
            if best_parse(match.group(0).lower()).normal_form != match.group(0).lower():
                stats.full_changed += 1
            if _is_term(parsed, terms):
                replacements[index] = _inflect(parsed, {"nomn", "sing"}) or parsed.normal_form
            elif parsed.tag.POS == "ADJF" and index + 1 < len(matches):
                following, noun = matches[index + 1], parses[index + 1]
                if text[match.end() : following.start()] == " " and _is_term(noun, terms):
                    gender = _noun_gender(noun)
                    lemma = _inflect(parsed, {"nomn", "sing", gender} if gender else {"nomn", "sing"})
                    if lemma:
                        replacements[index] = lemma
        pieces: list[str] = []
        cursor = 0
        for index, match in enumerate(matches):
            lemma = replacements.get(index)
            pieces.append(text[cursor : match.start()])
            surface = match.group(0)
            if lemma and lemma != surface.lower():
                new = _match_case(surface, lemma)
                self.audit.append(AuditEntry(chunk_id=chunk_id, rule="lemma", before=surface, after=new))
                stats.changed += 1
                pieces.append(new)
            else:
                pieces.append(surface)
            cursor = match.end()
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _english(self, chunk_id: str, text: str, stats: LemmaStats, terms: frozenset[str]) -> str:
        pieces: list[str] = []
        for paragraph in re.split(r"(\n\n)", text):
            if paragraph == "\n\n" or not LATIN_WORD.search(paragraph):
                pieces.append(paragraph)
                continue
            out: list[str] = []
            doc = english()(paragraph)
            for token in doc:
                surface = token.text
                if LATIN_WORD.fullmatch(surface):
                    stats.words += 1
                    if token.lemma_.lower() != surface.lower():
                        stats.full_changed += 1
                    singular = _regular_singular(surface.lower(), terms)
                    if token.tag_ in {"NNS", "NNPS"} and singular and not _verb_like(doc, token.i):
                        new = _match_case(surface, singular)
                        self.audit.append(AuditEntry(chunk_id=chunk_id, rule="lemma", before=surface, after=new))
                        stats.changed += 1
                        surface = new
                out.append(surface + token.whitespace_)
            pieces.append("".join(out))
        return "".join(pieces)


def _term_parse(word: str, terms: frozenset[str]) -> pymorphy3.analyzer.Parse:
    parses = morph().parse(word.lower())
    for parsed in parses:
        if parsed.tag.POS == "NOUN" and parsed.normal_form in terms and not parsed.tag.grammemes & NAME_TAGS:
            return parsed
    return best_parse(word.lower())


def _is_term(parsed: pymorphy3.analyzer.Parse, terms: frozenset[str]) -> bool:
    return parsed.tag.POS == "NOUN" and parsed.normal_form in terms and not parsed.tag.grammemes & NAME_TAGS


def _regular_singular(word: str, terms: frozenset[str]) -> str | None:
    candidates = [word[:-3] + "y"] if word.endswith("ies") else []
    candidates += [word[:-2], word[:-1]] if word.endswith("es") else [word[:-1]] if word.endswith("s") else []
    return next((c for c in candidates if c in terms), None)


def _inflect(parsed: pymorphy3.analyzer.Parse, grammemes: set[str]) -> str | None:
    inflected = parsed.inflect(grammemes)
    return inflected.word if inflected else None


def _noun_gender(noun: pymorphy3.analyzer.Parse) -> str | None:
    singular = noun.inflect({"nomn", "sing"})
    tag = singular.tag if singular else noun.tag
    return tag.gender


def _match_case(surface: str, lemma: str) -> str:
    if surface.isupper() and len(surface) > 1:
        return lemma.upper()
    if surface[:1].isupper():
        return lemma[:1].upper() + lemma[1:]
    return lemma


def _verb_like(doc: spacy.tokens.Doc, index: int) -> bool:
    previous = doc[index - 1].pos_ if index > 0 else ""
    following = doc[index + 1].pos_ if index + 1 < len(doc) else ""
    return previous in {"NOUN", "PROPN"} and following in {"ADP", "ADV", "DET", "PRON", "SCONJ", "PART", "NUM"}
