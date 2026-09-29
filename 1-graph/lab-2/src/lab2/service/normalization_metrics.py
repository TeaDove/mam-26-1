import difflib
import re
from dataclasses import dataclass
from datetime import date

from lab2.dto.models import AuditEntry, NormalizationMetrics
from lab2.service.normalization import Glossary
from lab2.util.text import DISPLAY_MATH, INLINE_MATH, WORD

NUMBER_PART = r"\d+(?:[.,]\d+)?(?:\s*[–\-÷]\s*\d+(?:[.,]\d+)?)?"
TEMPERATURE_ANY = re.compile(
    r"(?:\$\s*)?"
    + NUMBER_PART
    + r"\s*(?:\}\s*)?(?:<sup>\s*[0o°]\s*</sup>\s*[СC]|°\s*[СC]?|\^\s*\{?\s*(?:0|\\circ)\s*\}?\s*\$?\s*"
    r"(?:\\math\w+\s*\{\s*|\\bf\s*)?[СC]|\s?С(?=[)\s.,;]))(?!\s*/)"
)
TEMPERATURE_CANONICAL = re.compile(r"(?P<a>-?\d+(?:[.,]\d+)?)(?:–(?P<b>\d+(?:[.,]\d+)?))? °C")
OTHER_UNIT_ANY = re.compile(
    NUMBER_PART + r"\s*(?:\\?~?\s*\\mathrm\s*\{\s*H\s*/\s*M\s*M\s*\}|МПа|MPa|[NН]/мм2?²?|N/mm2?²?|kg/mm2?²?"
    r"|мм|mm|м\^?3?³?/ч\w*)"
)
OTHER_UNIT_CANONICAL = re.compile(r"\d+(?:[.,]\d+)?(?:–\d+(?:[.,]\d+)?)? (?:МПа|MPa|Н/мм²|N/mm²|kg/mm²|мм|mm|м³/ч)")
YEAR = re.compile(
    r"(?<![\d.,–-])(1[5-9]\d{2}|20\d{2})(?:s|-х|-е)?"
    r"(?![\d.,]*(?:\\ |\s|\$)*(?:°|%|мм|mm|K\b|С\b|MPa|МПа|м³|м\^|т\b|кг|kg))"
)
ANNOTATION_RULES = frozenset({"glossary_term", "abbreviation", "symbol_term"})
LATEX = re.compile(r"\\[a-zA-Z]+|[{}^_$]")


@dataclass
class NormalizationEvaluator:
    glossary: Glossary

    def evaluate(self, text: str, before: str | None, audit: list[AuditEntry]) -> NormalizationMetrics:
        words = WORD.findall(text)
        changed = _changed_words(before, text) if before is not None else None
        temperatures = [m.group(0) for m in TEMPERATURE_ANY.finditer(text)]
        canonical_temperatures = [t for t in temperatures if TEMPERATURE_CANONICAL.fullmatch(t)]
        in_range = [t for t in temperatures if _in_range(t)]
        other_units = [m.group(0) for m in OTHER_UNIT_ANY.finditer(text)]
        canonical_other = [u for u in other_units if OTHER_UNIT_CANONICAL.fullmatch(u)]
        units = len(temperatures) + len(other_units)
        years = [m.group(1) for m in YEAR.finditer(text)]
        valid_years = [y for y in years if 1800 <= int(y) <= date.today().year]
        found, with_english = self._glossary_coverage(text)
        annotations = [entry for entry in audit if entry.rule in ANNOTATION_RULES]
        return NormalizationMetrics(
            words=len(words),
            words_changed=changed,
            words_changed_share=round(changed / max(len(words), 1), 4) if changed is not None else None,
            annotation_events=len(annotations),
            distinct_terms_canonicalized=len({entry.after.split(" (")[0].casefold() for entry in annotations}),
            glossary_terms_found=found,
            glossary_terms_with_english=with_english,
            bridging_terms=None,
            unit_expressions=units,
            unit_expressions_canonical=len(canonical_temperatures) + len(canonical_other),
            unit_canonical_share=_share(len(canonical_temperatures) + len(canonical_other), units),
            temperatures=len(temperatures),
            temperatures_canonical_share=_share(len(canonical_temperatures), len(temperatures)),
            temperatures_in_range_share=_share(len(in_range), len(temperatures)),
            dates=len(years),
            dates_valid_share=_share(len(valid_years), len(years)),
        )

    def bridging_terms(self, texts: list[str]) -> int:
        lowered = [text.casefold() for text in texts]
        return sum(
            1
            for term in self.glossary.terms
            if all(re.search(rf"\b{re.escape(term.en.casefold())}\b", text) for text in lowered)
        )

    def _glossary_coverage(self, text: str) -> tuple[int, int]:
        found = with_english = 0
        lowered = text.casefold()
        for term in self.glossary.terms:
            english = re.search(rf"\b{re.escape(term.en.casefold())}\b", lowered) is not None
            russian = re.search(rf"\b{term.ru}", text, re.I) is not None
            if english or russian:
                found += 1
                with_english += english
        return found, with_english


def _share(part: int, total: int) -> float | None:
    return round(part / total, 4) if total else None


def _in_range(text: str) -> bool:
    values = [float(v.replace(",", ".")) for v in re.findall(r"\d+(?:[.,]\d+)?", LATEX.sub(" ", text))]
    return bool(values) and all(-273 <= v <= 1600 for v in values)


def _plain_words(text: str) -> list[str]:
    stripped = DISPLAY_MATH.sub(" ", text)
    stripped = INLINE_MATH.sub(" ", stripped)
    return [w for w in WORD.findall(stripped) if not re.fullmatch(r"[α-ωΑ-Ω]+", w)]


def _changed_words(before: str, after: str) -> int:
    old, new = _plain_words(before), _plain_words(after)
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag in ("replace", "delete", "insert")
    )
