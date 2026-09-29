from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

LANGUAGE_COLORS = {"en": "#2a78d6", "ru": "#eb6834", "both": "#1baf7a", "unknown": "#a3a29c"}
LANGUAGE_NAMES = {"en": "только английский текст", "ru": "только русский текст", "both": "оба текста"}
EDGE_COLOR = "#c9c8c2"
TEXT_COLOR = "#0b0b0b"


def draw_graph(graph: nx.Graph, title: str, target: Path, labels: int = 4) -> None:
    layout = nx.spring_layout(graph, seed=42, k=0.35, iterations=120)
    degrees = dict(graph.degree())
    figure, axis = plt.subplots(figsize=(8, 7), dpi=160)
    figure.patch.set_facecolor("white")
    nx.draw_networkx_edges(graph, layout, ax=axis, edge_color=EDGE_COLOR, width=0.6)
    for lang, color in LANGUAGE_COLORS.items():
        nodes = [n for n, d in graph.nodes(data=True) if d.get("lang") == lang]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            layout,
            nodelist=nodes,
            node_size=[14 + 9 * degrees[n] for n in nodes],
            node_color=color,
            edgecolors="white",
            linewidths=0.6,
            ax=axis,
        )
    top = sorted(degrees, key=lambda n: -degrees[n])[:labels]
    nx.draw_networkx_labels(
        graph,
        layout,
        labels={n: str(n)[:28] for n in top},
        font_size=8,
        font_color=TEXT_COLOR,
        ax=axis,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    axis.set_title(title, fontsize=12, color=TEXT_COLOR, loc="left")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, color=color, label=LANGUAGE_NAMES[lang])
        for lang, color in LANGUAGE_COLORS.items()
        if lang in LANGUAGE_NAMES and any(d.get("lang") == lang for _, d in graph.nodes(data=True))
    ]
    axis.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(target, facecolor="white")
    plt.close(figure)
