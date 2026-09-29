import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x5F, 0x5E, 0x5A)
CLEAN = RGBColor(0x2A, 0x78, 0xD6)
DIRTY = RGBColor(0xEB, 0x68, 0x34)
TINT = RGBColor(0xF1, 0xF4, 0xF8)
LINE = RGBColor(0xD9, 0xD8, 0xD3)
FONT = "Calibri"
NOISE_SHOWCASE = [
    "KONTROLIROVANNOY PROKATKI",
    "NEPRYEVNOLITOGO METALLA",
    "ТЕМПЕРАТУРА НА-ГРЕВА СЛЯ-БОВ",
    "СОРТАТОПРОКАТ",
    "САРАФАН",
    "ГРУББЛЕХ",
    "NIPPON KINZOKU GAKKAISHI",
    "TETSU-TO-HAGANÉ",
    "METALL. TRANS.",
]


def load(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def bfs_triple(summary: dict) -> str:
    return " / ".join(num(summary[f"bfs{depth}_mean_reached"], 0) for depth in (1, 2, 3))


def non_isolated(structure: dict) -> int:
    return structure["connected_components"] - structure["isolated_nodes"]


def pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank = self.prs.slide_layouts[6]

    def slide(self, title: str | None, notes: str) -> object:
        slide = self.prs.slides.add_slide(self.blank)
        if title:
            self.text(slide, title, 0.6, 0.45, 12.1, 0.8, size=28, bold=True)
        slide.notes_slide.notes_text_frame.text = notes
        return slide

    def text(
        self,
        slide: object,
        text: str | list[str],
        left: float,
        top: float,
        width: float,
        height: float,
        size: int = 16,
        bold: bool = False,
        color: RGBColor = INK,
        align: PP_ALIGN = PP_ALIGN.LEFT,
        font: str = FONT,
        spacing: int = 6,
    ) -> object:
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        lines = text if isinstance(text, list) else [text]
        for index, line in enumerate(lines):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.alignment = align
            paragraph.space_after = Pt(spacing)
            run = paragraph.add_run()
            run.text = line
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            run.font.name = font
        return box

    def card(self, slide: object, left: float, top: float, width: float, height: float) -> None:
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
        )
        shape.adjustments[0] = 0.06
        shape.fill.solid()
        shape.fill.fore_color.rgb = TINT
        shape.line.fill.background()
        shape.shadow.inherit = False

    def stat(self, slide: object, left: float, top: float, before: str, after: str, label: str) -> None:
        self.card(slide, left, top, 3.8, 1.75)
        box = slide.shapes.add_textbox(Inches(left + 0.3), Inches(top + 0.22), Inches(3.3), Inches(0.75))
        frame = box.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        paragraph = frame.paragraphs[0]
        for value, color, size in ((before, DIRTY, 26), ("  →  ", MUTED, 22), (after, CLEAN, 34)):
            run = paragraph.add_run()
            run.text = value
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = color
            run.font.name = FONT
        self.text(slide, label, left + 0.3, top + 1.0, 3.3, 0.7, size=13, color=MUTED)

    def table(
        self, slide: object, rows: list[list[str]], left: float, top: float, widths: list[float], size: int = 13
    ) -> None:
        height = 0.38 * len(rows)
        shape = slide.shapes.add_table(
            len(rows), len(widths), Inches(left), Inches(top), Inches(sum(widths)), Inches(height)
        )
        table = shape.table
        for column, width in enumerate(widths):
            table.columns[column].width = Inches(width)
        for r, row in enumerate(rows):
            table.rows[r].height = Inches(0.38)
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = TINT if r == 0 else RGBColor(0xFF, 0xFF, 0xFF)
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                frame = cell.text_frame
                frame.paragraphs[0].text = ""
                run = frame.paragraphs[0].add_run()
                run.text = value
                run.font.size = Pt(size)
                run.font.name = FONT
                run.font.bold = r == 0
                if r > 0 and c == len(row) - 1:
                    run.font.color.rgb = CLEAN
                elif r > 0 and c == len(row) - 2:
                    run.font.color.rgb = DIRTY
                else:
                    run.font.color.rgb = INK if r > 0 else MUTED
                frame.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT

    def bars(
        self,
        slide: object,
        title: str,
        categories: list[str],
        dirty: list[float],
        clean: list[float],
        left: float,
        top: float,
        width: float,
        height: float,
        number_format: str = "0",
    ) -> None:
        data = CategoryChartData()
        data.categories = categories
        data.add_series("грязный", dirty)
        data.add_series("чистый", clean)
        frame = slide.shapes.add_chart(
            XL_CHART_TYPE.BAR_CLUSTERED, Inches(left), Inches(top), Inches(width), Inches(height), data
        )
        chart = frame.chart
        chart.has_title = True
        chart.chart_title.text_frame.text = title
        title_run = chart.chart_title.text_frame.paragraphs[0].runs[0]
        title_run.font.size = Pt(14)
        title_run.font.bold = True
        title_run.font.color.rgb = INK
        chart.has_legend = True
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
        chart.legend.font.color.rgb = MUTED
        plot = chart.plots[0]
        plot.gap_width = 60
        plot.overlap = -10
        plot.has_data_labels = True
        labels = plot.data_labels
        labels.number_format = number_format
        labels.number_format_is_linked = False
        labels.position = XL_LABEL_POSITION.OUTSIDE_END
        labels.font.size = Pt(11)
        labels.font.color.rgb = INK
        for series, color in zip(plot.series, (DIRTY, CLEAN), strict=True):
            series.format.fill.solid()
            series.format.fill.fore_color.rgb = color
        category_axis = chart.category_axis
        category_axis.tick_labels.font.size = Pt(12)
        category_axis.tick_labels.font.color.rgb = INK
        category_axis.format.line.color.rgb = LINE
        category_axis.reverse_order = True
        value_axis = chart.value_axis
        value_axis.visible = False
        value_axis.has_major_gridlines = False

    def picture(self, slide: object, path: Path, left: float, top: float, height: float) -> None:
        slide.shapes.add_picture(str(path), Inches(left), Inches(top), height=Inches(height))

    def arrow_row(self, slide: object, label: str, steps: list[str], top: float, color: RGBColor) -> None:
        self.text(slide, label, 0.6, top + 0.12, 1.7, 0.5, size=16, bold=True, color=color)
        left = 2.35
        width = min(1.62, (10.4 - 0.25 * (len(steps) - 1)) / len(steps))
        for index, step in enumerate(steps):
            shape = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.72)
            )
            shape.adjustments[0] = 0.15
            shape.fill.solid()
            shape.fill.fore_color.rgb = TINT
            shape.line.color.rgb = color
            shape.line.width = Pt(1.25)
            shape.shadow.inherit = False
            frame = shape.text_frame
            frame.margin_left = frame.margin_right = Inches(0.04)
            paragraph = frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER
            run = paragraph.add_run()
            run.text = step
            run.font.size = Pt(12)
            run.font.color.rgb = INK
            run.font.name = FONT
            if index < len(steps) - 1:
                self.text(slide, "→", left + width, top + 0.14, 0.25, 0.4, size=16, color=MUTED, align=PP_ALIGN.CENTER)
            left += width + 0.25

    def save(self, path: Path) -> None:
        self.prs.save(str(path))


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "presentation.pptx"
    chunking = load("01_chunking.json")
    cleaning = load("02_cleaning.json")
    normalization = load("03_normalization.json")
    tokenization = load("04_tokenization.json")
    vectors = load("05_vectorization.json")
    comparison = load("07_graph_comparison.json")
    dirty, clean = comparison["dirty"], comparison["clean"]
    ds, cs = dirty["structure"], clean["structure"]
    deck = Deck()

    slide = deck.slide(
        None,
        "Тема: как предобработка текста меняет граф знаний. Корпус — два учебника по контролируемой прокатке "
        "стали: английский обзор Tanaka (1981) и русская статья. Оба текста строят один общий граф. "
        "Сравниваем граф из сырого текста MinerU и граф после полной предобработки.",
    )
    deck.text(slide, "Предобработка текста для графа знаний", 0.8, 2.0, 11.8, 1.0, size=36, bold=True)
    deck.text(
        slide,
        "Грязный и чистый граф по двум учебникам о контролируемой прокатке стали",
        0.8,
        3.25,
        11.8,
        0.6,
        size=22,
        color=MUTED,
    )
    deck.text(
        slide,
        "tanaka1981 (EN, 28 с.) + stat3 (RU, 10 с.) · MinerU → GraphRAG · gpt-4o-mini, bge-m3 · судья gpt-4.1",
        0.8,
        4.1,
        11.8,
        0.5,
        size=16,
        color=MUTED,
    )

    slide = deck.slide(
        "Эксперимент: отличие только в предобработке",
        "Оба графа строятся GraphRAG с одинаковыми настройками: gpt-4o-mini, эмбеддинги bge-m3, одни и те же промпты "
        "и типы сущностей, размер чанка 1200 токенов. Грязный граф — из markdown MinerU как есть. Чистый — из того же "
        "markdown после пяти этапов предобработки. После каждого этапа считаем метрики по каждой книге и суммарно, "
        "каждый этап проверял независимый ревьюер.",
    )
    deck.arrow_row(slide, "Грязный", ["PDF", "MinerU", "GraphRAG"], 1.8, DIRTY)
    deck.arrow_row(
        slide,
        "Чистый",
        ["PDF", "MinerU", "Chunking", "Очистка", "Нормали-\nзация", "Токени-\nзация", "Вектори-\nзация", "GraphRAG"],
        3.0,
        CLEAN,
    )
    deck.text(
        slide,
        [
            "Одинаковые: модели, промпты, типы сущностей, размер чанка 1200 токенов, кластеризация.",
            "Оба учебника → один граф. Метрики — после каждого этапа, по книгам и суммарно.",
            "Сравнение графов: структура, обходы (BFS/DFS/пути), покрытие ПрО, LLM-as-a-judge.",
        ],
        0.6,
        4.6,
        12.1,
        1.8,
        size=16,
        color=MUTED,
    )

    ch_d, ch_c = chunking["dirty_graphrag:total"], chunking["structural:total"]
    slide = deck.slide(
        "1. Chunking: по структуре, а не по счётчику токенов",
        "GraphRAG по умолчанию режет текст окнами по 1200 токенов — почти каждая граница рвёт предложение, а таблица "
        "разрезана пополам. Структурный разбивщик идёт по разделам и абзацам, не режет формулы и таблицы и "
        "предпочитает границы на конце предложения. Чанки сохранены — они пойдут в RAG.",
    )
    deck.stat(
        slide,
        0.6,
        1.55,
        pct(ch_d["boundaries_cutting_sentence_share"]),
        pct(ch_c["boundaries_cutting_sentence_share"]),
        "границ чанков режут предложение",
    )
    deck.stat(
        slide,
        0.6,
        3.5,
        str(ch_d["tables_broken"] + ch_d["formulas_broken"]),
        str(ch_c["tables_broken"] + ch_c["formulas_broken"]),
        "формул и таблиц разрезано границей",
    )
    deck.bars(
        slide,
        "Размер чанка, токенов o200k (среднее)",
        ["tanaka1981", "stat3", "всего"],
        [chunking[f"dirty_graphrag:{s}"]["tokens"]["mean"] for s in ("tanaka1981", "stat3", "total")],
        [chunking[f"structural:{s}"]["tokens"]["mean"] for s in ("tanaka1981", "stat3", "total")],
        4.9,
        1.45,
        7.8,
        3.9,
    )
    deck.text(
        slide,
        f"Чанков: {ch_d['n_chunks']} → {ch_c['n_chunks']}; "
        f"максимум {ch_c['tokens']['max']} токенов — GraphRAG не перерезает.",
        4.9,
        5.55,
        7.8,
        0.5,
        size=14,
        color=MUTED,
    )

    cl_d, cl_c = cleaning["dirty:total"], cleaning["clean:total"]
    slide = deck.slide(
        "2. Очистка: убираем шум, не теряя смысла",
        "Удаляем колонтитулы, номера страниц, ссылки на картинки, HTML-сноски, маркеры цитат, список литературы и "
        "метаданные авторов (PII). Склеиваем абзацы, разорванные страницами и подписями к рисункам, и переносы. "
        "Все правки журналируются. Числа, формулы и ячейки таблиц в содержательной части сохранены на 100%.",
    )
    deck.stat(slide, 0.6, 1.55, pct(cl_d["noise_share"], 1), pct(cl_c["noise_share"], 2), "доля шумовых символов")
    deck.stat(slide, 4.75, 1.55, str(cl_d["service_marks"]), str(cl_c["service_marks"]), "служебных меток")
    deck.stat(slide, 8.9, 1.55, str(cl_d["broken_paragraphs"]), str(cl_c["broken_paragraphs"]), "разорванных абзацев")
    deck.card(slide, 0.6, 3.65, 12.1, 2.9)
    deck.text(slide, "Было → стало", 0.9, 3.85, 5, 0.4, size=14, bold=True, color=MUTED)
    deck.text(
        slide,
        [
            "con-  ¶  International Metals Reviews  ¶  186  ¶  trolled   →   controlled",
            "In 1924 Arrowsmith $^{1}$ had …   →   In 1924 Arrowsmith had …",
            "ох- лаждение,  1 9 + 4 4  (в формуле)   →   охлаждение,  19 + 44",
            f"PII (авторы, аффилиации): {cl_d['pii_hits']} → {cl_c['pii_hits']};  "
            f"числа, формулы, ячейки таблиц: {pct(cl_c['numbers_preserved_share'])} / "
            f"{pct(cl_c['formulas_preserved_share'])} / {pct(cl_c['table_cells_preserved_share'])} сохранено",
        ],
        0.9,
        4.35,
        11.5,
        2.1,
        size=15,
        font="Consolas",
    )

    nm_r, nm_n = normalization["raw:total"], normalization["normalized:total"]
    slide = deck.slide(
        "3. Нормализация: единицы, аббревиатуры, термины",
        "Приводим единицы к одному виду (°C с пробелом, диапазоны через тире, Н/мм²), расшифровываем аббревиатуры и "
        "сводим термины к канону через двуязычный глоссарий. Важная находка: если писать английский термин в скобках "
        "после русского, gpt-4o-mini всё равно называет сущности по-русски. Работает только вариант, где канонический "
        "английский термин стоит первым — один раз на чанк. Так появились общие для двух книг узлы.",
    )
    deck.stat(
        slide,
        0.6,
        1.55,
        pct(nm_r["unit_canonical_share"]),
        pct(nm_n["unit_canonical_share"]),
        "единиц измерения в каноническом виде",
    )
    deck.stat(
        slide, 0.6, 3.5, str(nm_r["bridging_terms"]), str(nm_n["bridging_terms"]), "терминов глоссария в обеих книгах"
    )
    deck.card(slide, 4.75, 1.55, 7.95, 3.7)
    deck.text(slide, "stat3, было", 5.05, 1.75, 7.3, 0.4, size=14, bold=True, color=DIRTY)
    deck.text(
        slide,
        "…эффект КП, как способ измельчения зерна аустенита путём рекристаллизации…",
        5.05,
        2.15,
        7.4,
        0.8,
        size=15,
    )
    deck.text(slide, "стало", 5.05, 3.0, 7.3, 0.4, size=14, bold=True, color=CLEAN)
    deck.text(
        slide,
        "…эффект controlled rolling (КП, контролируемая прокатка), как способ grain refinement (измельчения зерна) "
        "austenite (аустенита) путём recrystallization (рекристаллизации)…",
        5.05,
        3.4,
        7.4,
        1.6,
        size=15,
    )
    deck.text(
        slide,
        "Английский в скобках после русского модель игнорирует — проверено на промпте GraphRAG.",
        4.75,
        5.45,
        7.95,
        0.5,
        size=14,
        color=MUTED,
    )

    tk_r, tk_n = tokenization["raw:total"], tokenization["normalized:total"]
    vc_d, vc_n = vectors["dirty_units:total"], vectors["normalized:total"]
    slide = deck.slide(
        "4–5. Токенизация и векторизация",
        "OOV считаем по словарю wordfreq плюс доменный глоссарий. Честная оговорка: у tanaka снижение OOV "
        "почти целиком "
        "дал удалённый список литературы, а у stat3 — склеенные переносы. Векторы bge-m3 (1024) посчитаны для каждого "
        "чанка — это база для следующего этапа, RAG, — и для каждой вершины графа, они записаны атрибутом вершин. "
        "Межъязыковое сходство выросло, но примерно 40% прироста дают вставленные английские термины.",
    )
    deck.table(
        slide,
        [
            ["Метрика", "грязный", "чистый"],
            ["OOV-rate (доля неизвестных слов)", pct(tk_r["oov_rate"], 1), pct(tk_n["oov_rate_original_words"], 1)],
            ["Уникальных OOV", str(tk_r["unique_oov"]), str(tk_n["unique_oov"])],
            [
                "Неразобранные $ (обрывки формул)",
                str(tk_r["unmatched_math_delimiters"]),
                str(tk_n["unmatched_math_delimiters"]),
            ],
            ["LaTeX-обозначения → текст (γ, °C, Ar3)", "—", str(tk_n["formulas_converted_to_text"])],
            ["Словарь TF-IDF, слов", str(vc_d["tfidf_vocabulary"]), str(vc_n["tfidf_vocabulary"])],
            ["Разреженность TF-IDF (окна по 300 слов)", num(vc_d["tfidf_sparsity"], 3), num(vc_n["tfidf_sparsity"], 3)],
            [
                "Сходство RU→EN, лучший чанк (cos)",
                num(vc_d["cross_language_best_match_cosine"], 3),
                num(vc_n["cross_language_best_match_cosine"], 3),
            ],
        ],
        0.6,
        1.55,
        [6.2, 2.0, 2.0],
        size=15,
    )
    deck.text(
        slide,
        [
            "Векторы bge-m3 (1024) — для чанков (задел для RAG) и атрибутом каждой вершины графа.",
            "Оговорки: OOV у tanaka упал в основном из-за удалённого списка литературы;",
            "около 40% роста межъязыкового сходства дают вставленные английские термины.",
        ],
        0.6,
        4.9,
        12.1,
        1.4,
        size=15,
        color=MUTED,
    )

    slide = deck.slide(
        "Графы: русская часть перестала быть островом",
        "Слева грязный граф: русский текст (оранжевый) — отдельный остров вокруг узла КП, между языками ноль рёбер. "
        "Справа чистый: русские сущности висят на общих узлах — CONTROLLED ROLLING, AUSTENITE, NIOBIUM и других; "
        "зелёные узлы извлечены из обеих книг.",
    )
    deck.picture(slide, FIGURES / "dirty_graph.png", 0.5, 1.35, 5.5)
    deck.picture(slide, FIGURES / "clean_graph.png", 6.85, 1.35, 5.5)
    deck.text(
        slide,
        f"рёбер EN–RU: {ds['edges_en_ru']}, общих узлов: {ds['nodes_both_languages']}",
        0.6,
        6.9,
        5.8,
        0.4,
        size=14,
        color=DIRTY,
        bold=True,
    )
    deck.text(
        slide,
        f"рёбер EN–RU: {cs['edges_en_ru']}, общих узлов: {cs['nodes_both_languages']}",
        6.95,
        6.9,
        5.8,
        0.4,
        size=14,
        color=CLEAN,
        bold=True,
    )

    slide = deck.slide(
        "Показатели графа",
        "Вершин стало меньше — ушёл шум (транслит, авторы, обрывки), а рёбер больше: граф плотнее. Компонент связности "
        "стало вдвое меньше, крупнейшая компонента охватывает больше вершин. Мостов почти столько же, а независимых "
        "циклов вдвое больше — у знаний появились альтернативные связи.",
    )
    deck.table(
        slide,
        [
            ["Показатель", "грязный", "чистый"],
            ["Вершины / рёбра", f"{ds['nodes']} / {ds['edges']}", f"{cs['nodes']} / {cs['edges']}"],
            [
                "Компоненты связности (без изолированных)",
                f"{ds['connected_components']} ({non_isolated(ds)})",
                f"{cs['connected_components']} ({non_isolated(cs)})",
            ],
            ["Доля крупнейшей компоненты", pct(ds["largest_component_share"]), pct(cs["largest_component_share"])],
            ["Изолированные вершины", str(ds["isolated_nodes"]), str(cs["isolated_nodes"])],
            [
                "Мосты / точки сочленения",
                f"{ds['bridges']} / {ds['articulation_points']}",
                f"{cs['bridges']} / {cs['articulation_points']}",
            ],
            ["Независимые циклы (E − V + C)", str(ds["cyclomatic_number"]), str(cs["cyclomatic_number"])],
            [
                "Средняя / макс. степень",
                f"{num(ds['degree_mean'])} / {ds['degree_max']}",
                f"{num(cs['degree_mean'])} / {cs['degree_max']}",
            ],
            ["Кластеризация", num(ds["average_clustering"], 3), num(cs["average_clustering"], 3)],
        ],
        0.6,
        1.55,
        [4.6, 1.6, 1.6],
        size=14,
    )
    types = ["process", "process parameter", "mechanical property", "microstructure", "phase", "person"]
    labels = ["процесс", "параметр", "мех. свойство", "микроструктура", "фаза", "персона"]
    deck.bars(
        slide,
        "Вершины по типам",
        labels,
        [ds["nodes_by_type"].get(t, 0) for t in types],
        [cs["nodes_by_type"].get(t, 0) for t in types],
        8.7,
        1.45,
        4.2,
        5.2,
    )

    dt, ct = dirty["traversal"]["summary"], clean["traversal"]["summary"]
    paths = [
        p
        for p in clean["traversal"]["pairs"]
        if p.get("path")
        and not any(
            q["source"] == p["source"] and q["target"] == p["target"] and q.get("path")
            for q in dirty["traversal"]["pairs"]
        )
    ]
    path = max(paths, key=lambda p: len(p["path"]), default=None)
    slide = deck.slide(
        "Обходы графа: BFS, DFS, кратчайшие пути",
        "Берём 8 ключевых концептов как стартовые вершины и 10 пар концептов из разных частей предметной области. "
        "В грязном графе связаны только 2 пары из 10 — остальные в разных компонентах, в основном из-за языкового "
        "разрыва. В чистом связаны все 10. BFS на 2 шага теперь достаёт узлы из другого языка, а доля шумовых "
        "узлов среди достигнутых — по оценке судьи.",
    )
    deck.stat(
        slide,
        0.6,
        1.55,
        f"{dt['pairs_connected']}/{dt['pairs_total']}",
        f"{ct['pairs_connected']}/{ct['pairs_total']}",
        "пар концептов связаны путём",
    )
    deck.stat(
        slide,
        4.75,
        1.55,
        pct(dirty["coverage"]["domain_coverage"]),
        pct(clean["coverage"]["domain_coverage"], 1),
        "покрытие ПрО: 40 концептов найдено",
    )
    deck.stat(
        slide,
        8.9,
        1.55,
        pct(dt["bfs2_other_language_share"]),
        pct(ct["bfs2_other_language_share"]),
        "узлов другого языка в BFS-2",
    )
    deck.table(
        slide,
        [
            ["Обход от 8 ключевых концептов", "грязный", "чистый"],
            [
                "BFS-1 / BFS-2 / BFS-3, узлов в среднем",
                bfs_triple(dt),
                bfs_triple(ct),
            ],
            ["Концептов ПрО в BFS-2", num(dt["bfs2_mean_concepts"], 1), num(ct["bfs2_mean_concepts"], 1)],
            ["Шумовых узлов в BFS-2 (судья)", pct(dt["bfs2_noise_share"], 1), pct(ct["bfs2_noise_share"], 1)],
            ["DFS: достижимо узлов", num(dt["dfs_mean_reached"], 0), num(ct["dfs_mean_reached"], 0)],
            ["Средняя длина пути", num(dt["pairs_mean_length"] or 0, 1), num(ct["pairs_mean_length"] or 0, 1)],
        ],
        0.6,
        3.65,
        [5.2, 2.4, 2.4],
        size=14,
    )
    if path:
        deck.text(slide, "Пример пути в чистом графе", 0.6, 6.15, 6.0, 0.4, size=13, bold=True, color=MUTED)
        deck.text(slide, "  →  ".join(path["path"]), 0.6, 6.5, 12.1, 0.5, size=14)

    ju_d, ju_c = dirty["judge_units"], clean["judge_units"]
    jn_d, jn_c = dirty["judge_nodes"], clean["judge_nodes"]
    du_d, du_c = dirty["duplicates"], clean["duplicates"]
    slide = deck.slide(
        "LLM-as-a-judge (gpt-4.1)",
        "Судья — модель сильнее экстрактора. Он оценивал каждый текстовый фрагмент вместе с извлечёнными из него "
        "сущностями и связями, отдельно размечал все вершины как основные, второстепенные или шум и подтверждал "
        "кандидатов в дубликаты, найденных по близости эмбеддингов названий.",
    )
    deck.table(
        slide,
        [
            ["Критерий", "грязный", "чистый"],
            ["Полнота по тексту (1–5)", num(ju_d["completeness_mean"]), num(ju_c["completeness_mean"])],
            ["Покрытие понятий фрагмента (1–5)", num(ju_d["domain_coverage_mean"]), num(ju_c["domain_coverage_mean"])],
            [
                "Целостность формул и таблиц (1–5)",
                num(ju_d["formula_table_integrity_mean"]),
                num(ju_c["formula_table_integrity_mean"]),
            ],
            ["Противоречия тексту, шт.", str(ju_d["contradictions_total"]), str(ju_c["contradictions_total"])],
            [
                "Шумовые сущности во фрагментах",
                pct(ju_d["noise_entities_share"], 1),
                pct(ju_c["noise_entities_share"], 1),
            ],
            ["Шумовые вершины графа", pct(jn_d["noise_share"], 1), pct(jn_c["noise_share"], 1)],
            ["Корреференции: дубликаты вершин", str(du_d["confirmed"]), str(du_c["confirmed"])],
            ["  из них межъязыковые", str(du_d["confirmed_cross_language"]), str(du_c["confirmed_cross_language"])],
        ],
        0.6,
        1.55,
        [5.4, 1.9, 1.9],
        size=15,
    )
    deck.card(slide, 10.15, 1.55, 2.6, 3.9)
    deck.text(slide, "Шум грязного графа", 10.4, 1.75, 2.2, 0.4, size=13, bold=True, color=MUTED)
    examples = [n for n in NOISE_SHOWCASE if n in dirty["noise_examples"]][:8]
    deck.text(slide, examples, 10.4, 2.2, 2.2, 3.2, size=11, spacing=3)

    slide = deck.slide(
        "Выводы",
        "Главное: предобработка сделала из двух изолированных графов один. Без неё русская часть — отдельный остров, "
        "а обход графа не может пройти из одного учебника в другой. Ограничения: глоссарий составлен вручную под "
        "этот корпус; один прогон GraphRAG без повторов; судья — одна модель. Следующий шаг — RAG на сохранённых "
        "чанках и векторах.",
    )
    deck.text(
        slide,
        [
            "Chunking по структуре почти перестал рвать предложения и не режет формулы и таблицы.",
            "Очистка убрала почти весь шум, сохранив все числа, формулы и таблицы содержательной части.",
            "Нормализация терминов «английский канон первым» связала английскую и русскую части графа.",
            "Чистый граф связнее: компонент меньше, циклов больше, все пары концептов достижимы.",
            "Не решено: дубликаты вида ед./мн. число остались — нужна лемматизация сущностей.",
            "Ограничения: ручной глоссарий, один прогон, один судья. Дальше — RAG на чанках и векторах.",
        ],
        0.6,
        1.6,
        12.0,
        4.5,
        size=20,
        spacing=16,
    )
    deck.save(target)
    print(target)


if __name__ == "__main__":
    main()
