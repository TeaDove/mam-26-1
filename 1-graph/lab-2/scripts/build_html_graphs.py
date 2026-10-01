import html
import re
from pathlib import Path

import networkx as nx
from pyvis.network import Network

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "results" / "graphs"
LANGUAGE_COLORS = {"en": "#4a3aa7", "ru": "#eda100", "both": "#1baf7a", "unknown": "#a3a29c"}
LANGUAGE_NAMES = {"en": "английский текст", "ru": "русский текст", "both": "оба текста", "unknown": "неизвестно"}
TITLES = {
    "dirty": "Грязный граф: текст MinerU без списка литературы",
    "clean": "Чистый граф: после предобработки",
}


def tooltip(name: str, data: dict, degree: int) -> str:
    description = html.escape(str(data.get("description", ""))[:400])
    return (
        f"<b>{html.escape(name)}</b><br>"
        f"тип: {html.escape(str(data.get('type', '')))}<br>"
        f"источник: {LANGUAGE_NAMES.get(data.get('lang', 'unknown'), '')}<br>"
        f"степень: {degree}<br><br>{description}"
    )


def header(arm: str, graph: nx.Graph) -> str:
    present = {data.get("lang", "unknown") for _, data in graph.nodes(data=True)}
    chips = "".join(
        f'<span style="margin-right:18px"><span style="display:inline-block;width:11px;height:11px;'
        f'border-radius:50%;background:{color};margin-right:6px"></span>{LANGUAGE_NAMES[lang]}</span>'
        for lang, color in LANGUAGE_COLORS.items()
        if lang in present
    )
    stats = f"{graph.number_of_nodes()} вершин, {graph.number_of_edges()} рёбер · размер узла — степень"
    return (
        '<div style="font-family:Arial,sans-serif;color:#1f2328;padding:12px 16px 4px">'
        f'<div style="font-size:20px;font-weight:bold">{TITLES[arm]}</div>'
        f'<div style="font-size:13px;color:#5f5e5a;margin:6px 0">{stats}</div>'
        f'<div style="font-size:13px">{chips}</div></div>'
    )


def build(arm: str) -> Path:
    graph = nx.read_graphml(GRAPHS / f"{arm}_graph_with_vectors.graphml.gz")
    network = Network(height="92vh", width="100%", bgcolor="#ffffff", font_color="#1f2328", cdn_resources="in_line")
    for name, data in graph.nodes(data=True):
        degree = graph.degree(name)
        network.add_node(
            name,
            label=name if degree >= 6 else "",
            title=tooltip(name, data, degree),
            color=LANGUAGE_COLORS.get(data.get("lang", "unknown"), LANGUAGE_COLORS["unknown"]),
            size=6 + 2.5 * degree**0.8,
        )
    for source, target, data in graph.edges(data=True):
        network.add_edge(
            source,
            target,
            title=html.escape(str(data.get("description", ""))[:300]),
            color="#c9c8c2",
            width=1,
        )
    network.barnes_hut(gravity=-6000, spring_length=110, damping=0.4)
    target = GRAPHS / f"{arm}_graph.html"
    network.write_html(str(target), notebook=False)
    page = re.sub(r"<center>\s*<h1>\s*</h1>\s*</center>", "", target.read_text(encoding="utf-8"))
    page = page.replace("<body>", "<body>\n" + header(arm, graph), 1)
    target.write_text(page, encoding="utf-8")
    return target


def main() -> None:
    for arm in ("dirty", "clean"):
        print(build(arm))


if __name__ == "__main__":
    main()
