import re
from dataclasses import dataclass, field
from functools import cache

import pymorphy3
import spacy
from spacy.language import Language

from lab2.dto.models import AuditEntry

CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
LATIN_WORD = re.compile(r"[A-Za-z]+")
PLACEHOLDER = re.compile(r"\d+")
NOUN_TAGS = frozenset({"NOUN"})
ADJECTIVE_TAGS = frozenset({"ADJF", "PRTF"})
PLURAL_TAGS = frozenset({"NNS", "NNPS"})


@cache
def morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@cache
def english() -> Language:
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


@dataclass
class LemmaStats:
    words: int = 0
    changed: int = 0
    full_changed: int = 0


@dataclass
class Lemmatizer:
    audit: list[AuditEntry] = field(default_factory=list)
    stats: dict[str, LemmaStats] = field(default_factory=dict)

    def lemmatize(self, chunk_id: str, text: str, lang: str) -> str:
        stats = self.stats.setdefault(chunk_id, LemmaStats())
        if lang == "ru":
            return self._russian(chunk_id, text, stats)
        return self._english(chunk_id, text, stats)

    def _russian(self, chunk_id: str, text: str, stats: LemmaStats) -> str:
        matches = list(CYRILLIC_WORD.finditer(text))
        parses = [morph().parse(m.group(0))[0] for m in matches]
        replacements: dict[int, str] = {}
        for index, (match, parsed) in enumerate(zip(matches, parses, strict=True)):
            stats.words += 1
            if parsed.normal_form != match.group(0).lower():
                stats.full_changed += 1
            if parsed.tag.POS in NOUN_TAGS:
                lemma = _inflect(parsed, {"nomn", "sing"}) or parsed.normal_form
                replacements[index] = lemma
            elif parsed.tag.POS in ADJECTIVE_TAGS and index + 1 < len(matches):
                following, noun = matches[index + 1], parses[index + 1]
                adjacent = text[match.end() : following.start()] == " "
                if adjacent and noun.tag.POS in NOUN_TAGS:
                    target = _noun_gender(noun)
                    lemma = _inflect(parsed, {"nomn", "sing", target} if target else {"nomn", "sing"})
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

    def _english(self, chunk_id: str, text: str, stats: LemmaStats) -> str:
        pieces: list[str] = []
        for paragraph in re.split(r"(\n\n)", text):
            if paragraph == "\n\n" or not LATIN_WORD.search(paragraph):
                pieces.append(paragraph)
                continue
            doc = english()(paragraph)
            out: list[str] = []
            for token in doc:
                surface = token.text
                if LATIN_WORD.fullmatch(surface) and not PLACEHOLDER.search(surface):
                    stats.words += 1
                    if token.lemma_.lower() != surface.lower():
                        stats.full_changed += 1
                    if token.tag_ in PLURAL_TAGS and token.lemma_.lower() != surface.lower() and len(surface) > 3:
                        new = _match_case(surface, token.lemma_.lower())
                        self.audit.append(AuditEntry(chunk_id=chunk_id, rule="lemma", before=surface, after=new))
                        stats.changed += 1
                        surface = new
                out.append(surface + token.whitespace_)
            pieces.append("".join(out))
        return "".join(pieces)


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
