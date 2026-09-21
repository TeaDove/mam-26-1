# Lab 2 — MinerU + GraphRAG knowledge graphs

Two textbooks on controlled rolling of steel were parsed with MinerU and turned into
knowledge graphs with GraphRAG:

- `tanaka1981/` — *Controlled rolling of steel* (English, 28 pages);
- `stat3/` — a Russian paper on controlled rolling (10 pages).

See `SOURCES.md` for the download links and checksums of the original PDFs. The PDFs
themselves are kept outside the repository.

## Pipeline

1. **Parsing — MinerU 4.0.5**, run locally on a GPU host (RTX 5060, 8 GB). Each PDF was
   converted to Markdown (`tanaka1981.md`, `stat3.md`), including layout, formulas and
   table recognition.
2. **Indexing — GraphRAG 3.1.2** with `gpt-4o-mini` as the completion model, reached
   through the project HTTPS proxy. Text embeddings are computed **locally** by a
   `bge-m3` (fp16, CUDA) server exposing an OpenAI-compatible `/v1` endpoint, so no
   embedding traffic leaves the host.
3. **Visualisation** — `output/graph.graphml` is rendered to an interactive
   `graph.html` (pyvis, physics enabled, node size by degree, tooltips with entity type
   and description) and a static `graph.png` (matplotlib spring layout, the 25
   highest-degree nodes labelled).

### Entity types

Both projects extract the same metallurgy-oriented entity types:

`material`, `steel grade`, `alloying element`, `process`, `process parameter`,
`mechanical property`, `microstructure`, `phase`, `equipment`, `phenomenon`, `person`,
`organization`.

### Network tuning

The HTTPS path to the proxy stalls on TCP flows larger than roughly 20 KB in either
direction. Three settings in `settings.yaml` keep every request below that limit:

- `call_args.extra_headers.Connection: close` — one request per TCP connection;
- `call_args.timeout: 90` plus a bounded `retry` block — a stalled call is aborted and
  retried instead of hanging forever;
- `community_reports.max_input_length: 1200` and `max_length: 1200` — the prompts for
  top-level communities would otherwise reach about 22 KB and never complete.

Embeddings avoid the proxy entirely by running locally.

## Results

| Metric | tanaka1981 | stat3 |
| --- | --- | --- |
| Entities | 312 | 79 |
| Relationships | 287 | 79 |
| Communities | 51 | 17 |
| Community reports | 51 | 17 |
| Text units | 26 | 7 |
| Graph nodes | 237 | 74 |
| Graph edges | 282 | 78 |

The graph node count is lower than the entity count because isolated entities — those
for which no relationship was extracted — are not part of the exported graph.

The dominant entity in `tanaka1981` is `CONTROLLED ROLLING` (degree 36), followed by
`CONTROLLED-ROLLED STEEL`, `AUSTENITE` and `RECRYSTALLIZATION`. In `stat3` the hub is
`КП` (controlled rolling, degree 20) together with `КОНТРОЛИРОВАННАЯ ПРОКАТКА` and
`КПУО`. Per-book numbers and the top-15 entities by degree are in each `stats.md`.

## Contents of each book folder

| File | Description |
| --- | --- |
| `graph.graphml` | GraphRAG entity/relationship graph |
| `graph.html` | Interactive pyvis visualisation |
| `graph.png` | Static overview of the graph |
| `stats.md` | Counts and the top-15 entities by degree |
| `entities.csv` | Exported entities (title, type, degree, frequency, description) |
| `relationships.csv` | Exported relationships (source, target, weight, description) |
| `<book>.md` | MinerU Markdown parse of the PDF |
| `settings.yaml` | GraphRAG configuration used for the run (API key as a placeholder) |
