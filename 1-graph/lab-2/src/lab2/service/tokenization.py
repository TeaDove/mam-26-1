import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from pydantic import BaseModel
from tokenizers import Tokenizer
from wordfreq import zipf_frequency

from lab2.dto.models import Chunk, TokenizationMetrics
from lab2.util.text import CITATION, DISPLAY_MATH, INLINE_MATH, count_tokens, encoder, formulas

TOKEN = re.compile(
    r"(?P<markup><[^>\n]+>|!\[[^\]]*\]\([^)]*\))"
    r"|(?P<formula>\$\$.*?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$)"
    r"|(?P<number>\d+(?:[.,]\d+)*(?:–\d+(?:[.,]\d+)*)?)"
    r"|(?P<word>[^\W\d_]+(?:[-'][^\W\d_]+)*)"
    r"|(?P<symbol>[^\w\s])",
    re.S,
)
GREEK = re.compile(r"^[α-ωΑ-Ω]+$")
MIXED_SCRIPT = re.compile(r"(?=.*[a-zA-Z])(?=.*[а-яА-ЯёЁ])")
FORMULA_CONVERSION_RULES = frozenset({"greek_symbol", "temperature_math", "math_ocr", "critical_point"})


class TokenizedChunk(BaseModel):
    chunk_id: str
    book: str
    words: list[str]
    formulas: list[str]
    numbers: list[str]
    o200k_tokens: int
    bge_tokens: int


@cache
def bge_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


@dataclass
class TextTokenizer:
    bge_tokenizer_path: Path
    domain_lexicon: frozenset[str]
    domain_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    def tokenize(self, chunk: Chunk) -> TokenizedChunk:
        words: list[str] = []
        found_formulas: list[str] = []
        numbers: list[str] = []
        for match in TOKEN.finditer(chunk.text):
            if match.group("formula"):
                if not CITATION.fullmatch(match.group("formula")):
                    found_formulas.append(match.group("formula"))
            elif match.group("number"):
                numbers.append(match.group("number"))
            elif match.group("word"):
                words.append(match.group("word"))
        return TokenizedChunk(
            chunk_id=chunk.chunk_id,
            book=chunk.book,
            words=words,
            formulas=found_formulas,
            numbers=numbers,
            o200k_tokens=count_tokens(chunk.text),
            bge_tokens=len(self._bge().encode(chunk.text, add_special_tokens=False).ids),
        )

    def words(self, text: str) -> list[str]:
        return [m.group("word") for m in TOKEN.finditer(text) if m.group("word")]

    def is_oov(self, word: str) -> bool:
        lowered = word.casefold()
        if lowered in self.domain_lexicon or GREEK.match(word):
            return False
        if any(pattern.fullmatch(word) for pattern in self.domain_patterns):
            return False
        if MIXED_SCRIPT.match(word):
            return True
        lang = "ru" if re.search(r"[а-яё]", lowered) else "en"
        return any(zipf_frequency(part, lang) == 0 and part not in self.domain_lexicon for part in lowered.split("-"))

    def metrics(
        self, texts: list[str], chunk_texts: list[str], inserted_words: int, converted_formulas: int
    ) -> TokenizationMetrics:
        words = [word for text in texts for word in self.words(text)]
        oov = Counter(word.casefold() for word in words if self.is_oov(word))
        oov_total = sum(oov.values())
        all_formulas = [f for text in texts for f in formulas(text)]
        unk_id = self._bge().token_to_id("<unk>")
        bge_ids = [token for text in texts for token in self._bge().encode(text, add_special_tokens=False).ids]
        o200k = sum(count_tokens(text) for text in texts)
        word_counts = Counter(words)
        o200k_words = sum(len(encoder().encode(" " + w)) * n for w, n in word_counts.items())
        bge_words = sum(
            len(self._bge().encode(" " + w, add_special_tokens=False).ids) * n for w, n in word_counts.items()
        )
        count = max(len(words), 1)
        original = max(len(words) - inserted_words, 1)
        return TokenizationMetrics(
            words=len(words),
            words_inserted_by_normalization=inserted_words,
            unique_words=len({word.casefold() for word in words}),
            oov_words=oov_total,
            oov_rate=round(oov_total / count, 4),
            oov_rate_original_words=round(oov_total / original, 4),
            unique_oov=len(oov),
            oov_examples=[word for word, _ in oov.most_common(12)],
            o200k_tokens=o200k,
            o200k_tokens_per_word_whole_text=round(o200k / count, 3),
            o200k_word_fertility=round(o200k_words / count, 3),
            bge_tokens=len(bge_ids),
            bge_tokens_per_word_whole_text=round(len(bge_ids) / count, 3),
            bge_word_fertility=round(bge_words / count, 3),
            bge_unk_tokens=sum(1 for token in bge_ids if token == unk_id),
            formulas=len(all_formulas),
            formulas_converted_to_text=converted_formulas,
            formulas_balanced_share=round(sum(1 for f in all_formulas if _balanced(f)) / max(len(all_formulas), 1), 4),
            unmatched_math_delimiters=sum(_unmatched_math(text) for text in chunk_texts),
        )

    def _bge(self) -> Tokenizer:
        return bge_tokenizer(str(self.bge_tokenizer_path))


def _balanced(formula: str) -> bool:
    body = re.sub(r"\\[{}]", "", formula)
    depth = 0
    for char in body:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth < 0:
            return False
    begins = len(re.findall(r"\\begin\{", body))
    ends = len(re.findall(r"\\end\{", body))
    lefts = len(re.findall(r"\\left\b", body))
    rights = len(re.findall(r"\\right\b", body))
    return depth == 0 and begins == ends and lefts == rights


def _unmatched_math(text: str) -> int:
    without_display = DISPLAY_MATH.sub(" ", text)
    display_leftover = without_display.count("$$")
    without_inline = INLINE_MATH.sub(" ", without_display.replace("$$", " "))
    return display_leftover + without_inline.count("$")
