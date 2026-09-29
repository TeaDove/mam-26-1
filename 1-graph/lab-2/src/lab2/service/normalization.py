import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from lab2.dto.models import AuditEntry, Chunk
from lab2.util.text import CYRILLIC, DISPLAY_MATH, INLINE_MATH, TABLE, count_tokens

MATH_WRAPPERS = re.compile(r"\\(?:mathrm|mathbf|bf|text|scriptstyle|boldsymbol|rm)\b")
TEMPERATURE_MATH = re.compile(
    r"^(?P<pre>\(?(?:с|c)?)(?P<approx>\\sim)?(?P<num>\d+(?:[.,]\d+)?(?:[-–]\d+(?:[.,]\d+)?)?)"
    r"\^(?:0|\\circ|o)(?P<unit>C(?:/c)?)(?P<post>[,.)]*)$"
)
GREEK_MATH = re.compile(r"\$\s*((?:\\(?:gamma|alpha|delta)\s*[-+,]?\s*)+)\$")
HEADER_UNIT = re.compile(r"(\$[^$\n]*?),\s*\^\{?\s*0\s*\}?\s*C\$")


class Replacement(BaseModel):
    source: str
    target: str


class PatternReplacement(BaseModel):
    pattern: str
    target: str


class Abbreviation(BaseModel):
    short: str
    long: str
    en: str


class Term(BaseModel):
    en: str
    ru: str
    symbol: str | None = None


class Glossary(BaseModel):
    math_replacements: list[Replacement]
    critical_points: list[PatternReplacement]
    greek: dict[str, str]
    unit_replacements: list[PatternReplacement]
    abbreviations: list[Abbreviation]
    terms: list[Term]

    @classmethod
    def load(cls, path: Path) -> Glossary:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Normalizer:
    glossary: Glossary
    audit: list[AuditEntry] = field(default_factory=list)

    def normalize(self, chunks: list[Chunk]) -> list[Chunk]:
        result: list[Chunk] = []
        for chunk in chunks:
            text = self._normalize_text(chunk.chunk_id, chunk.text, chunk.lang)
            result.append(chunk.model_copy(update={"text": text, "n_tokens": count_tokens(text)}))
        return result

    def _normalize_text(self, chunk_id: str, text: str, lang: str) -> str:
        for item in self.glossary.math_replacements:
            text = self._literal(chunk_id, "math_ocr", item.source, item.target, text)
        for item in self.glossary.critical_points:
            text = self._regex(chunk_id, "critical_point", re.compile(item.pattern), item.target, text)
        text = self._regex(chunk_id, "unit", HEADER_UNIT, r"\1$, °C", text)
        text = INLINE_MATH.sub(lambda m: self._temperature_math(chunk_id, m.group(0)), text)
        text = GREEK_MATH.sub(lambda m: self._greek(chunk_id, m), text)
        text = self._outside_math(text, lambda piece: self._units(chunk_id, piece))
        masked, protected = _mask(text)
        masked = self._annotate_terms(chunk_id, masked, lang)
        masked = self._expand_abbreviations(chunk_id, masked)
        return _unmask(masked, protected)

    def _temperature_math(self, chunk_id: str, formula: str) -> str:
        plain = MATH_WRAPPERS.sub("", formula.strip("$"))
        plain = re.sub(r"[{}\s~]|\\ ", "", plain).replace("\\,", "")
        match = TEMPERATURE_MATH.match(plain)
        if not match:
            return formula
        pre = match.group("pre").replace("c", "с ").replace("с", "с ").replace("  ", " ")
        approx = "~" if match.group("approx") else ""
        number = match.group("num").replace("-", "–")
        unit = "°C/с" if match.group("unit") == "C/c" else "°C"
        after = f"{pre}{approx}{number} {unit}{match.group('post')}".replace("( ", "(")
        self._log(chunk_id, "temperature_math", formula, after)
        return after

    def _greek(self, chunk_id: str, match: re.Match[str]) -> str:
        content = match.group(1)
        for name, symbol in self.glossary.greek.items():
            content = re.sub(rf"\\{name}\b", symbol, content)
        after = re.sub(r"\s*-\s*", "–", re.sub(r"\s+", " ", content.strip()))
        self._log(chunk_id, "greek_symbol", match.group(0), after)
        return after

    def _units(self, chunk_id: str, text: str) -> str:
        for item in self.glossary.unit_replacements:
            text = self._regex(chunk_id, "unit", re.compile(item.pattern), item.target, text)
        return text

    def _expand_abbreviations(self, chunk_id: str, text: str) -> str:
        for abbreviation in sorted(self.glossary.abbreviations, key=lambda a: -len(a.short)):
            pattern = re.compile(rf"(?<![\w(+]){re.escape(abbreviation.short)}(?![\w+])(?! \()")
            match = pattern.search(text)
            if not match:
                continue
            if not CYRILLIC.search(abbreviation.short):
                after = f"{abbreviation.short} ({abbreviation.long})"
            elif abbreviation.long:
                after = f"{abbreviation.en} ({abbreviation.short}, {abbreviation.long})"
            else:
                after = f"{abbreviation.en} ({abbreviation.short})"
            self._log(chunk_id, "abbreviation", abbreviation.short, after)
            text = text[: match.start()] + after + text[match.end() :]
        return text

    def _annotate_terms(self, chunk_id: str, text: str, lang: str) -> str:
        for term in self.glossary.terms:
            if lang == "ru":
                text = self._annotate_first(chunk_id, text, re.compile(rf"\b{term.ru}", re.I), term.en)
            if term.symbol:
                symbol_pattern = re.compile(
                    rf"(?<![\w$\\-])(?>{re.escape(term.symbol)}(?:-[a-z]{{2,}})?)(?![\w$(])(?![–-]\s)(?!-[A-Z])(?! \()"
                )
                text = self._annotate_symbol(chunk_id, text, symbol_pattern, term)
        return text

    def _annotate_first(self, chunk_id: str, text: str, pattern: re.Pattern[str], english: str) -> str:
        match = next((m for m in pattern.finditer(text) if not _inside_parentheses(text, m.start())), None)
        if not match:
            return text
        after = f"{english} ({match.group(0)})"
        self._log(chunk_id, "glossary_term", match.group(0), after)
        return text[: match.start()] + after + text[match.end() :]

    def _annotate_symbol(self, chunk_id: str, text: str, pattern: re.Pattern[str], term: Term) -> str:
        match = pattern.search(text)
        if not match:
            return text
        surface = match.group(0)
        suffix = surface[len(term.symbol or "") :]
        english = term.en + suffix.replace("-", " ")
        after = f"{surface} ({english})"
        self._log(chunk_id, "symbol_term", surface, after)
        return text[: match.start()] + after + text[match.end() :]

    def _outside_math(self, text: str, function: Callable[[str], str]) -> str:
        pieces: list[str] = []
        cursor = 0
        for match in DISPLAY_MATH.finditer(text):
            pieces.extend([self._outside_inline(text[cursor : match.start()], function), match.group(0)])
            cursor = match.end()
        pieces.append(self._outside_inline(text[cursor:], function))
        return "".join(pieces)

    def _outside_inline(self, text: str, function: Callable[[str], str]) -> str:
        pieces: list[str] = []
        cursor = 0
        for match in INLINE_MATH.finditer(text):
            pieces.extend([function(text[cursor : match.start()]), match.group(0)])
            cursor = match.end()
        pieces.append(function(text[cursor:]))
        return "".join(pieces)

    def _literal(self, chunk_id: str, rule: str, source: str, target: str, text: str) -> str:
        count = text.count(source)
        for _ in range(count):
            self._log(chunk_id, rule, source, target)
        return text.replace(source, target)

    def _regex(self, chunk_id: str, rule: str, pattern: re.Pattern[str], target: str, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            after = match.expand(target)
            if after != match.group(0):
                self._log(chunk_id, rule, match.group(0), after)
            return after

        return pattern.sub(replace, text)

    def _log(self, chunk_id: str, rule: str, before: str, after: str) -> None:
        self.audit.append(AuditEntry(chunk_id=chunk_id, rule=rule, before=before, after=after))


def _mask(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def keep(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"{len(protected) - 1}"

    for pattern in (DISPLAY_MATH, TABLE, INLINE_MATH):
        text = pattern.sub(keep, text)
    return text, protected


def _unmask(text: str, protected: list[str]) -> str:
    previous = None
    while previous != text:
        previous, text = text, re.sub(r"(\d+)", lambda m: protected[int(m.group(1))], text)
    return text


def _inside_parentheses(text: str, position: int) -> bool:
    window = text[max(position - 80, 0) : position]
    return window.rfind("(") > window.rfind(")")
