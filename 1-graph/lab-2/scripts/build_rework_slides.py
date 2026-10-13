import sys
from pathlib import Path

from build_presentation import METRICS, MUTED, ROOT, Deck, Metrics, load, num, pct
from pydantic import TypeAdapter

from lab2.dto.models import (
    AdjudicatedVertex,
    ConsistencyVerdict,
    Coreference,
    GraphComparison,
    GraphReport,
    JudgeMatrix,
    VariantComparison,
    VariantMetric,
)

ADJUDICATION = ROOT / "data" / "judge_adjudication.json"
REFERENCE_RUNS = ("08_consistency_{}.json", "09_consistency_gpt-4-1-run2_{}.json")
LOCAL_RUNS = ("09_consistency_qwen2-5-14b-instruct_{}.json", "09_consistency_qwen2-5-32b-instruct_{}.json")
JUDGES = (
    ("gpt-4.1", "gpt-4.1", "gpt-4-1-run2", "gpt-4.1"),
    ("Qwen2.5-14B", "qwen2-5-14b-instruct", "qwen2-5-14b-instruct-run2", "qwen2.5:14b-instruct"),
    ("Qwen2.5-32B", "qwen2-5-32b-instruct", None, "qwen2.5:32b-instruct"),
    ("Gemma3-12B", "gemma3-12b", None, None),
)


def fixes_rows(m: Metrics) -> list[list[str]]:
    d, c = m.dirty, m.clean
    return [
        ["Замечание", "Что сделали", "Результат: грязный → чистый", "Вывод"],
        [
            "1. Шум вырос на 2 п. п. — проверить разметку LaTeX-формул",
            "Разметка исправлена: пропускала Т_0, Г_Θ, SIGMA_D, DELTA_P, ловила элементы NB, TI. "
            "Формулы — в отдельный подграф",
            f"формул {d.formulas.vertices} и {c.formulas.vertices}; шум было 5,2% → 7,5%, стало "
            f"{pct(d.noise.noise_share, 1)} → {pct(c.noise.noise_share, 1)}",
            "Гипотеза подтвердилась: рост шума давали формулы",
        ],
        [
            "2. Понизить cos — корреференция должна упасть",
            "Перебор порога cos 0,95…0,80; корреференция разделена на три вида: формы слова, один язык, переводы RU–EN",
            f"переводы (cos ≥ 0,85) {d.coreference.translation_pairs} → {c.coreference.translation_pairs}; один "
            f"язык (≥ 0,92) {d.coreference.same_language_pairs} → {c.coreference.same_language_pairs}; формы "
            f"слова {d.coreference.lemma_duplicate_vertices} → {c.coreference.lemma_duplicate_vertices}",
            "Частично: переводы упали; ниже 0,9 cos ловит родственные понятия, а не дубликаты",
        ],
        [
            "3. Почему выросли независимые циклы",
            "Профиль циклов: на 100 вершин, длины, через хаб, через вершины обеих книг",
            f"{d.cycles.cycles} → {c.cycles.cycles}; на 100 вершин {num(d.cycles.cycles_per_100_vertices, 1)} → "
            f"{num(c.cycles.cycles_per_100_vertices, 1)}; через обе книги {d.cycles.cross_book_cycles} → "
            f"{c.cycles.cross_book_cycles}",
            "Рост пропорционален размеру графа; новые циклы сшивают книги",
        ],
    ]


def fixes_slide(deck: Deck, m: Metrics) -> None:
    d, c = m.dirty, m.clean
    slide = deck.slide(
        "Работа над ошибками: три замечания",
        "Первое: шум вырос на 2 п. п. Проверили разметку формул регулярными выражениями — она пропускала "
        "обозначения с кириллическим или словесным греческим индексом и ловила химические элементы. Исправили и "
        f"вынесли вершины-формулы в отдельный подграф: {d.formulas.vertices} в грязном и {c.formulas.vertices} в "
        f"чистом графе. Без них шум {pct(d.noise.noise_share, 1)} → {pct(c.noise.noise_share, 1)}, то есть рост "
        "давали именно формулы. Второе: понизили порог косинуса. Дубликаты-переводы при 0,85 действительно "
        f"падают — {d.coreference.translation_pairs} → {c.coreference.translation_pairs}, — но общее число пар "
        "растёт: ниже 0,9 косинус ловит не дубликаты, а родственные понятия, и в чистом графе их больше. Поэтому "
        "корреференцию считаем по видам с разными порогами. Третье: циклов больше ровно пропорционально размеру "
        f"графа — {num(d.cycles.cycles_per_100_vertices, 1)} и {num(c.cycles.cycles_per_100_vertices, 1)} на 100 "
        f"вершин; новые циклы проходят через вершины обеих книг: {d.cycles.cross_book_cycles} → "
        f"{c.cycles.cross_book_cycles}.",
    )
    deck.table(slide, fixes_rows(m), 0.6, 1.4, [2.4, 4.1, 3.2, 2.4], size=13, value_columns=(2, 2), row_height=1.15)
    deck.text(
        slide,
        "Остальные критерии пересчитаны и не изменились; ответы судьи взяты из прошлого прогона.",
        0.6,
        6.75,
        12.1,
        0.4,
        size=13,
        color=MUTED,
    )


def sweep(report: GraphReport, field: str) -> list[float]:
    coreference: Coreference = report.coreference
    return [getattr(point, field) for point in coreference.threshold_sweep]


def evidence_slide(deck: Deck, m: Metrics) -> None:
    d, c = m.dirty, m.clean
    thresholds = [num(p.threshold) for p in d.coreference.threshold_sweep]
    slide = deck.slide(
        "Корреференция по порогу cos и что внутри",
        "Слева — дубликаты-переводы: пары «русское название — английское» по порогу косинуса. В чистом графе их "
        "меньше при любом пороге до 0,85; ниже начинаются ложные пары вроде «ТЕМПЕРАТУРА = HEATING TEMPERATURE». "
        "Справа — пары на одном языке: с понижением порога их число растёт в обоих графах, но в чистом быстрее. "
        "Примеры при 0,85–0,88: RECRYSTALLIZATION = STATIC RECRYSTALLIZATION, CONTROLLED ROLLING = "
        "LOW-TEMPERATURE CONTROLLED ROLLING — это общее и частное понятие, а не дубликат. Нормализация сделала "
        "названия вершин английскими и подробными, поэтому таких близких пар больше.",
    )
    deck.bars(
        slide,
        "Переводы RU–EN, пар",
        thresholds,
        sweep(d, "translation_pairs"),
        sweep(c, "translation_pairs"),
        0.5,
        1.3,
        6.1,
        4.4,
    )
    deck.bars(
        slide,
        "Один язык, пар",
        thresholds,
        sweep(d, "same_language_pairs"),
        sweep(c, "same_language_pairs"),
        6.75,
        1.3,
        6.1,
        4.4,
    )
    deck.text(
        slide,
        [
            "Переводы при 0,85: "
            + "; ".join(item.rsplit(" (", 1)[0] for item in d.coreference.translation_examples)
            + "@D",
            "Подграф формул чистого графа крепится к " + ", ".join(c.formulas.anchors[:2]) + "@D",
        ],
        0.6,
        5.9,
        12.1,
        0.9,
        size=12,
        color=MUTED,
        spacing=4,
    )


def kappa(matrix: JudgeMatrix, left: str, right: str, arm: str) -> float:
    return next(
        pair.contradiction_kappa
        for pair in matrix.pairs
        if pair.arm == arm and {pair.left, pair.right} == {left, right}
    )


def flags(name: str, arm: str) -> list[bool]:
    path = METRICS / name.format(arm)
    verdicts = TypeAdapter(list[ConsistencyVerdict]).validate_json(path.read_text(encoding="utf-8"))
    return [verdict.contradictions > 0 for verdict in verdicts]


def corrected_share(adjudicated: list[AdjudicatedVertex], arm: str) -> float:
    reference = [flags(name, arm) for name in REFERENCE_RUNS]
    local = [flags(name, arm) for name in LOCAL_RUNS]
    rows = list(zip(*reference, *local, strict=True))
    gpt_only = sum(all(row[:2]) and not any(row[2:]) for row in rows)
    local_only = sum(not any(row[:2]) and all(row[2:]) for row in rows)

    def real_share(disagreement: str) -> float:
        cases = [a for a in adjudicated if a.arm == arm and a.disagreement == disagreement]
        return sum(a.has_contradiction for a in cases) / len(cases)

    positive = sum(reference[0]) - gpt_only
    positive += gpt_only * real_share("only gpt-4.1 flags") + local_only * real_share("only local models flag")
    return positive / len(rows)


def judge_rows(matrix: JudgeMatrix, adjudicated: list[AdjudicatedVertex]) -> list[list[str]]:
    runs = {run.label: run for run in matrix.runs}
    rows = [["Судья", "Противоречий", "κ с gpt-4.1", "Повтор, κ", "Верно на 25"]]
    for title, label, repeat, flag_key in JUDGES:
        run = runs[label]
        agreement = (
            "—"
            if label == "gpt-4.1"
            else " / ".join(num(kappa(matrix, "gpt-4.1", label, arm)) for arm in ("dirty", "clean"))
        )
        stability = " / ".join(num(kappa(matrix, label, repeat, arm)) for arm in ("dirty", "clean")) if repeat else "—"
        correct = sum((a.judge_flags[flag_key] if flag_key else False) == a.has_contradiction for a in adjudicated)
        rows.append(
            [
                title,
                f"{pct(run.dirty_contradiction_share, 1)} → {pct(run.clean_contradiction_share, 1)}",
                agreement,
                stability,
                f"{correct} из {len(adjudicated)}",
            ]
        )
    return rows


def judge_slide(deck: Deck, m: Metrics) -> None:
    matrix = load("10_judge_matrix.json", JudgeMatrix)
    adjudicated = TypeAdapter(list[AdjudicatedVertex]).validate_json(ADJUDICATION.read_text(encoding="utf-8"))
    dirty, clean = (corrected_share(adjudicated, arm) for arm in ("dirty", "clean"))
    slide = deck.slide(
        "Какой судья: gpt-4.1 или локальная модель",
        "Те же 120 вершин каждого графа оценили gpt-4.1 дважды и три локальные модели через Ollama. Больше — не "
        "лучше: Qwen2.5-32B согласна с gpt-4.1 хуже, чем 14B, а на грязном графе находит вдвое меньше "
        "противоречий. Gemma3-12B не нашла ни одного. Локальные модели детерминированы — повтор Qwen-14B совпал "
        "один в один, — но обе Qwen считают чистый граф хуже грязного: это систематический сдвиг, а не шум. 25 вершин, "
        "где gpt-4.1 и обе Qwen расходятся, разметили вслепую по тексту: ни один судья там не надёжен, gpt-4.1 "
        "чуть точнее. С поправкой на разметку gpt-4.1 даёт "
        f"{pct(dirty, 1)} → {pct(clean, 1)} — направление то же. Оставляем судьёй gpt-4.1; разница графов по "
        "непротиворечивости — на границе точности судьи.",
    )
    deck.table(
        slide,
        judge_rows(matrix, adjudicated),
        0.6,
        1.4,
        [2.6, 3.0, 2.6, 2.0, 1.9],
        size=14,
        value_columns=(1, 1),
        row_height=0.5,
    )
    deck.text(
        slide,
        [
            "Больше параметров не помогает: 32B согласна с gpt-4.1 хуже, чем 14B, и на грязном графе находит "
            "вдвое меньше противоречий",
            "Gemma3-12B не находит противоречий вообще; «всегда нет» на спорных вершинах угадывает не хуже Qwen",
            "Обе Qwen ставят чистый граф ниже грязного — систематический сдвиг, а не шум",
            f"Слепая разметка 25 спорных вершин: с поправкой gpt-4.1 даёт {pct(dirty, 1)} → {pct(clean, 1)}. "
            "Судья — gpt-4.1",
        ],
        0.6,
        4.3,
        12.1,
        2.4,
        size=15,
        spacing=8,
    )
    deck.text(
        slide,
        "Противоречий — доля вершин, грязный → чистый; κ — каппа Коэна по флагу «есть противоречие», грязный / "
        "чистый; «Верно на 25» — совпадение со слепой разметкой спорных вершин",
        0.6,
        6.85,
        12.1,
        0.4,
        size=12,
        color=MUTED,
    )


VARIANT_LABELS = {
    "input tokens": "Токенов на входе",
    "vertices": "Вершин",
    "edges": "Рёбер",
    "connected components": "Компонент связности",
    "largest component share": "Крупнейшая компонента",
    "RU vertices in largest component": "Русских вершин в ней",
    "cycles per 100 vertices": "Циклов на 100 вершин",
    "completeness (semantic)": "Полнота по тексту",
    "word-form duplicate vertices": "Дубликатов форм слова",
    "vertices with contradictions": "Вершин с противоречиями",
}


def variant_value(metric: VariantMetric, value: float) -> str:
    if metric.kind == "share":
        return pct(value, 1)
    return num(value, 1) if value % 1 else f"{int(value):,}".replace(",", " ")


def variant_rows(variants: VariantComparison) -> list[list[str]]:
    rows = [["Чистый граф", "база", "повтор", "пересечение", "явно"]]
    by_name = {metric.name: metric for metric in variants.metrics}
    for name, label in VARIANT_LABELS.items():
        metric = by_name[name]
        rows.append(
            [
                label,
                variant_value(metric, metric.baseline),
                variant_value(metric, metric.rerun),
                variant_value(metric, metric.overlap),
                "да" if metric.clear else "",
            ]
        )
    base, overlap = variants.baseline_retrieval, variants.overlap_retrieval
    rows.append(["MRR поиска по чанкам", num(base.mrr), "—", num(overlap.mrr), ""])
    return rows


def overlap_slide(deck: Deck, m: Metrics) -> None:
    variants = load("11_variants.json", VariantComparison)
    overlap = load("11_overlap/08_graph_comparison.json", GraphComparison).clean
    base, retrieval = m.clean, variants.overlap_retrieval
    hub_share = overlap.cycles.cycles_through_top_hub / overlap.cycles.cycles
    slide = deck.slide(
        "Чанки с пересечением 15%: граф связнее, но беднее",
        "Каждый чанк получил по 7,5% соседей слева и справа: целые предложения, формулы и таблицы целиком, не "
        "больше 1200 токенов, только внутри книги. Базовые чанки те же — меняется один фактор. Чтобы отличить "
        "эффект от случайности GraphRAG, тот же чистый вход проиндексировали второй раз — это столбец «повтор». "
        "Отклонение явное, если оно больше удвоенного шума и не меньше 3 п. п. или 10%. Экстрактор берёт с чанка "
        "примерно одно и то же число сущностей независимо от длины, поэтому часть его внимания уходит на повтор "
        "соседнего текста: уникальных вершин меньше, связей между ними больше. Граф связнее, русская книга "
        f"пришита к хабу {overlap.cycles.top_hub}, но рост циклов в основном звёздный, через хаб. Полнота и "
        f"поиск хуже: MRR {num(m.retrieval['clean_structural:all'].mrr)} → {num(retrieval.mrr)}, соседние чанки "
        "стали похожи друг на друга. Непротиворечивость не изменилась.",
    )
    deck.table(
        slide,
        variant_rows(variants),
        0.6,
        1.4,
        [3.0, 1.3, 1.3, 1.6, 0.8],
        size=13,
        value_columns=(3, 3),
        row_height=0.42,
    )
    deck.text(
        slide,
        [
            "С чанка — около 15 сущностей при любой длине: пересечение тратит их на повтор соседа",
            "Уникальных вершин меньше, связей больше — граф связнее, русская книга в общей компоненте",
            f"{pct(hub_share)} циклов идут через хаб {overlap.cycles.top_hub} — рост звёздный (в базе "
            f"{pct(base.cycles.cycles_through_top_hub / base.cycles.cycles)})",
            "Полнота и поиск хуже: соседние чанки стали похожи, редкие факты тонут",
            "Вывод: пересечение — компромисс; для RAG оставляем чанки без пересечения",
        ],
        9.0,
        1.4,
        3.9,
        5.2,
        size=14,
        spacing=10,
    )
    deck.text(
        slide,
        "повтор — тот же вход, проиндексированный второй раз, показывает шум GraphRAG; «явно» — отклонение больше "
        "удвоенного шума",
        0.6,
        6.85,
        12.1,
        0.4,
        size=12,
        color=MUTED,
    )


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "rework_2026-10-12.pptx"
    metrics = Metrics.load_all()
    deck = Deck()
    for build in (fixes_slide, evidence_slide, judge_slide, overlap_slide):
        build(deck, metrics)
    deck.save(target)
    print(target)


if __name__ == "__main__":
    main()
