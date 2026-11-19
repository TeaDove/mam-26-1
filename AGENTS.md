# Agent rules for this repository

## Version control

- Commit and push directly to `master`. Do not create or use any other branches.
- Do not commit large files. If a large file appears (models, datasets, caches, build
  artifacts, big PDFs/images), add it to `.gitignore` instead of committing it.
- Do not commit personal data (names, surnames, contacts, IDs, etc.). The homework
  solutions themselves are fine to commit. Bibliographic references to published works
  (textbook and paper authors) are not considered personal data.

## Repository layout

Course directory names use `<semester>-<subject>`: the leading number is the semester, not a course index.
Use `1-` for first-semester subjects and keep the directory tree current when adding or renaming folders.
The tree lists tracked directories; local environments, caches, and ignored artifacts are omitted.
Course names below are English translations of the titles in the
[official timetable](https://misis.ru/students/schedule/).

```text
.
├── 1-algorithms/ — Algorithms and Data Structures
│   ├── homework/ — Homework notebooks
│   └── lectures/ — Lecture materials
└── 1-graph/ — Graph Theory and Graph Neural Networks
    ├── lab-1/ — Knowledge graph construction
    │   ├── graph/ — Graph exports and statistics
    │   └── sources/ — Source texts
    ├── lab-2/ — Text preprocessing and knowledge graph comparison
    │   ├── configs/ — Experiment configuration
    │   ├── data/ — Input and intermediate datasets
    │   │   ├── 00_dirty/ — Unprocessed source texts
    │   │   ├── gold/ — Reference annotations
    │   │   │   └── pages/ — Annotated reference pages
    │   │   └── raw/ — Original extracted texts
    │   ├── graphrag-clean/ — GraphRAG workspace for preprocessed texts
    │   │   ├── input/ — GraphRAG input documents
    │   │   ├── output/ — GraphRAG index exports
    │   │   └── prompts/ — GraphRAG prompt templates
    │   ├── graphrag-dirty/ — GraphRAG workspace for unprocessed texts
    │   │   ├── input/ — GraphRAG input documents
    │   │   ├── output/ — GraphRAG index exports
    │   │   └── prompts/ — GraphRAG prompt templates
    │   ├── results/ — Experiment results
    │   │   ├── figures/ — Figures
    │   │   ├── graphs/ — Graph visualizations and exports
    │   │   ├── metrics/ — Evaluation metrics
    │   │   │   ├── 11_overlap/ — Results with overlapping chunks
    │   │   │   └── 11_rerun/ — Repeated evaluation results
    │   │   └── v1/ — Initial experiment results
    │   │       └── metrics/ — Evaluation metrics
    │   ├── scripts/ — Experiment utilities
    │   ├── src/ — Source code
    │   │   └── lab2/ — Text preprocessing and evaluation package
    │   │       ├── api/ — Command-line interface
    │   │       ├── dto/ — Data transfer objects
    │   │       ├── service/ — Experiment orchestration
    │   │       ├── supplier/ — External service integrations
    │   │       └── util/ — Shared utilities
    │   └── tests/ — Tests
    └── lab-3/ — Retrieval-augmented generation with graph neural networks
        ├── configs/ — Experiment configuration
        ├── results/ — Experiment results
        │   ├── answers/ — Generated answers
        │   ├── before_cosine_training/ — Results before cosine-decoder training
        │   │   ├── answers/ — Generated answers
        │   │   └── metrics/ — Evaluation metrics
        │   ├── graph_only_v1/ — Initial graph-only retrieval results
        │   │   └── answers/ — Generated answers
        │   └── metrics/ — Evaluation metrics
        │       └── ragas/ — Ragas evaluation reports
        ├── scripts/ — Experiment utilities
        ├── src/ — Source code
        │   └── lab3/ — Graph neural network retrieval package
        └── tests/ — Tests
```
