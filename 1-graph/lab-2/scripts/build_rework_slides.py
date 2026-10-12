import sys
from pathlib import Path

from build_presentation import MUTED, ROOT, Deck, Metrics, num, pct

from lab2.dto.models import Coreference, GraphReport


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


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "rework_2026-10-12.pptx"
    metrics = Metrics.load_all()
    deck = Deck()
    for build in (fixes_slide, evidence_slide):
        build(deck, metrics)
    deck.save(target)
    print(target)


if __name__ == "__main__":
    main()
