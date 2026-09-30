import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

import networkx as nx
import numpy as np
from wordfreq import zipf_frequency

from lab2.service.lemmatization import english, morph
from lab2.util.text import DISPLAY_MATH, TABLE, table_cells

GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "sigma": "σ",
    "Sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "theta": "θ",
    "mu": "μ",
    "kappa": "κ",
    "pi": "π",
}
PRONOUNS = frozenset(
    {
        "он",
        "она",
        "оно",
        "они",
        "его",
        "её",
        "ее",
        "их",
        "ему",
        "ей",
        "им",
        "это",
        "этот",
        "эта",
        "эти",
        "тот",
        "та",
        "те",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "he",
        "she",
        "which",
    }
)
UNIT = r"(?:°\s*[CС]|%|мм|mm|µm|μm|MPa|МПа|N/mm2|Н/мм2|N/mm²|Н/мм²|kg/mm2|K|с|s|min|ч|h)"
NOISE_RULES: dict[str, re.Pattern[str]] = {
    "punctuation_only": re.compile(r"^[\W_]+$"),
    "formula_variable": re.compile(r"^(?:[A-Za-zΑ-Ωα-ω]{1,2}\d*|.*[\\$^{}].*|[A-Za-zΑ-Ωα-ω]_\S+)$"),
    "number_or_unit": re.compile(rf"^[~≈<>≤≥]?\s*[\d.,\s–÷+-]+\s*{UNIT}?$|^{UNIT}$", re.I),
    "pronoun_or_function_word": re.compile(r"^(?:" + "|".join(sorted(PRONOUNS)) + r")$", re.I),
    "broken_word": re.compile(r"(?:\b[А-ЯA-Z]{1,4}-[А-ЯA-Z]{2,}\b.*-|\b\w+-\s|\s-\w)", re.I),
}


def window(graph: nx.Graph, node: str, depth: int = 2, breadth: int = 3) -> list[str]:
    visited = [node]
    frontier = [node]
    for _ in range(depth):
        following: list[str] = []
        for current in frontier:
            neighbours = sorted(
                (n for n in graph.neighbors(current) if n not in visited),
                key=lambda n, c=current: -float(graph[c][n].get("weight", 1.0)),
            )[:breadth]
            for neighbour in neighbours:
                visited.append(neighbour)
                following.append(neighbour)
        frontier = following
    return visited


def window_text(graph: nx.Graph, nodes: list[str]) -> str:
    parts = [f"{n} {graph.nodes[n].get('description', '')}" for n in nodes]
    members = set(nodes)
    parts.extend(
        str(data.get("description", ""))
        for a, b, data in graph.edges(nodes, data=True)
        if a in members and b in members
    )
    return " ".join(parts)


@cache
def lemma_key(text: str) -> str:
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+|\d+", text.replace("ё", "е"))
    lemmas: list[str] = []
    for word in words:
        if re.match(r"[А-Яа-яЁё]", word):
            lemmas.append(morph().parse(word.lower())[0].normal_form)
        elif word.isdigit():
            lemmas.append(word)
        else:
            lemmas.append(_english_lemma(word.lower()))
    return " ".join(lemmas)


@cache
def _english_lemma(word: str) -> str:
    return english()(word)[0].lemma_.lower()


def noise_labels(name: str, vocabulary: frozenset[str], allowed: frozenset[str]) -> list[str]:
    stripped = name.strip()
    if stripped.casefold() in allowed:
        return []
    labels = [rule for rule, pattern in NOISE_RULES.items() if pattern.search(stripped)]
    if _unknown_word(stripped, vocabulary, allowed):
        labels.append("unknown_word")
    return labels


def _unknown_word(name: str, vocabulary: frozenset[str], allowed: frozenset[str]) -> bool:
    for word in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", name):
        lowered = word.lower()
        if re.search(r"[a-z]", lowered) and re.search(r"[а-я]", lowered):
            return True
        if lowered in vocabulary or lowered in allowed:
            continue
        lang = "ru" if re.search(r"[а-я]", lowered) else "en"
        if zipf_frequency(lowered, lang) == 0 and zipf_frequency(lemma_key(lowered), lang) == 0:
            return True
    return False


def source_vocabulary(texts: list[str]) -> frozenset[str]:
    words: set[str] = set()
    for text in texts:
        words.update(word.lower() for word in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text))
    return frozenset(words)


def formula_atoms(latex: str) -> set[str]:
    text = latex.strip("$").replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\tag\{\d+\}", " ", text)
    for name, symbol in GREEK.items():
        text = re.sub(rf"\\{name}\b", symbol, text)
    text = re.sub(r"\\(?:mathrm|mathbf|text|bf|rm|boldsymbol|scriptstyle|quad|begin|end|array)\b", " ", text)
    text = re.sub(r"\{r l\}|[{}\s]", "", text)
    atoms = set(re.findall(r"[A-Za-zΑ-Ωα-ω]_?[A-Za-z0-9]{0,4}", text.replace("_", "")))
    atoms |= set(re.findall(r"\d+(?:\.\d+)?", text))
    return {a.lower() for a in atoms if len(a) > 1 or a in {"d"}}


def plain_atoms(text: str) -> str:
    flat = text
    for name, symbol in GREEK.items():
        flat = re.sub(rf"\\{name}\b|\b{name}\b", symbol, flat, flags=re.I)
    return re.sub(r"[\s{}_$\\]", "", flat).lower()


def formulas_and_tables(source: str) -> tuple[list[str], list[list[str]]]:
    formulas = DISPLAY_MATH.findall(source)
    tables = [[c for c in table_cells(t) if re.search(r"\d", c)] for t in TABLE.findall(source)]
    return formulas, tables


@dataclass
class IntegrityResult:
    in_vertex: int
    in_window: int
    broken: int
    best_share: list[float]


def integrity(graph: nx.Graph, items: list[set[str]], threshold: float = 0.6) -> IntegrityResult:
    vertex_texts = {n: plain_atoms(f"{n} {graph.nodes[n].get('description', '')}") for n in graph.nodes}
    window_texts = {n: plain_atoms(window_text(graph, window(graph, n))) for n in graph.nodes}
    in_vertex = in_window = broken = 0
    shares: list[float] = []
    for atoms in items:
        if not atoms:
            continue
        best_vertex = max((_share(atoms, t) for t in vertex_texts.values()), default=0.0)
        best_window = max((_share(atoms, t) for t in window_texts.values()), default=0.0)
        shares.append(round(best_window, 3))
        if best_vertex >= threshold:
            in_vertex += 1
        elif best_window >= threshold:
            in_window += 1
        else:
            broken += 1
    return IntegrityResult(in_vertex=in_vertex, in_window=in_window, broken=broken, best_share=shares)


def _share(atoms: set[str], text: str) -> float:
    return sum(1 for atom in atoms if atom in text) / len(atoms)


def lift_pronouns(graph: nx.Graph, depth: int = 3) -> tuple[int, int]:
    work = graph.copy()
    lifted = subtree = 0
    for node in [n for n in work.nodes if lemma_key(str(n)) in PRONOUNS]:
        neighbours = list(work.neighbors(node))
        parent = max(neighbours, key=work.degree, default=None)
        blocked = work.subgraph(n for n in work.nodes if n != parent)
        subtree += len(nx.single_source_shortest_path_length(blocked, node, cutoff=depth)) - 1
        for child in neighbours:
            if parent is not None and child != parent:
                work.add_edge(parent, child, weight=1.0)
        work.remove_node(node)
        lifted += 1
    return lifted, subtree


def duplicate_groups(names: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for name in names:
        key = lemma_key(name)
        if key:
            groups.setdefault(key, []).append(name)
    return [group for group in groups.values() if len(group) > 1]


def embedding_duplicates(names: list[str], vectors: np.ndarray, threshold: float) -> list[tuple[str, str, float]]:
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if similarity[i, j] >= threshold and lemma_key(names[i]) != lemma_key(names[j]):
                pairs.append((names[i], names[j], round(float(similarity[i, j]), 3)))
    return pairs


def sample_nodes(graph: nx.Graph, must: list[str], size: int, seed: int = 42) -> list[str]:
    candidates = sorted((n for n in graph.nodes if graph.degree(n) > 0 and n not in must), key=str)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = list(dict.fromkeys(must))[:size]
    return chosen + candidates[: max(size - len(chosen), 0)]


def rule_counts(labels: dict[str, list[str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for values in labels.values():
        counter.update(values)
    return dict(counter.most_common())


def text_terms(text: str, lang: str, limit: int) -> list[str]:
    counts: Counter[str] = Counter()
    if lang == "ru":
        words = re.findall(r"[А-Яа-яЁё]+", text)
        parses = [morph().parse(word.lower())[0] for word in words]
        for index, parsed in enumerate(parses):
            if parsed.tag.POS == "NOUN" and len(words[index]) >= 5:
                counts[parsed.normal_form] += 1
            if index + 1 < len(parses) and parsed.tag.POS in {"ADJF", "PRTF"} and parses[index + 1].tag.POS == "NOUN":
                counts[f"{parsed.normal_form} {parses[index + 1].normal_form}"] += 1
    else:
        for paragraph in text.split("\n\n"):
            doc = english()(re.sub(r"\$[^$]*\$", " ", paragraph))
            run: list[str] = []
            for token in [*doc, None]:
                if token is not None and token.pos_ in {"ADJ", "NOUN", "PROPN"} and token.is_alpha:
                    run.append(token.lemma_.lower())
                    continue
                while run and run[-1] and len(run) > 0 and not _is_noun_lemma(run[-1]):
                    run.pop()
                if len(run) >= 2:
                    counts[" ".join(run[-3:])] += 1
                if len(run) == 1 and len(run[0]) >= 5:
                    counts[run[0]] += 1
                run = []
    terms = [
        term
        for term, count in counts.most_common()
        if (count >= 2 if " " in term else count >= 4) and not set(term.split()) & PRONOUNS
    ]
    return terms[:limit]


@cache
def _is_noun_lemma(word: str) -> bool:
    return english()(word)[0].pos_ in {"NOUN", "PROPN"}
