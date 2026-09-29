# tanaka1981 + stat3 (combined graph) — GraphRAG statistics

| Metric | Value |
| --- | --- |
| Entities | 465 |
| Relationships | 402 |
| Communities | 90 |
| Community reports | 90 |
| Text units | 33 |
| Graph nodes | 340 |
| Graph edges | 388 |
| Density | 0.0067 |

## Graph structure

| Metric | Value |
| --- | --- |
| Connected components | 28 |
| Largest component (nodes) | 217 |
| Largest component share of nodes | 63.8% |
| Components of size 2 | 15 |
| Isolated nodes in graph | 0 |
| Bridges | 224 |
| Self-loops | 0 |
| Cyclomatic number (E - V + C) | 76 |
| Independent cycles (cycle basis) | 76 |
| Average clustering coefficient | 0.0546 |

## Degree statistics

| Metric | Value |
| --- | --- |
| Min degree | 1 |
| Max degree | 52 |
| Mean degree | 2.28 |
| Median degree | 1 |
| Nodes with degree 1 (leaves) | 211 |

## Component sizes

| Size | Components |
| --- | --- |
| 217 | 1 |
| 47 | 1 |
| 7 | 1 |
| 6 | 1 |
| 5 | 1 |
| 4 | 4 |
| 3 | 4 |
| 2 | 15 |

## Nodes by entity type

| Type | Nodes |
| --- | --- |
| PROCESS | 61 |
| PROCESS PARAMETER | 50 |
| PERSON | 41 |
| PHENOMENON | 31 |
| MECHANICAL PROPERTY | 29 |
| MICROSTRUCTURE | 28 |
| PHASE | 24 |
| ALLOYING ELEMENT | 18 |
| STEEL GRADE | 16 |
| ORGANIZATION | 15 |
| MATERIAL | 14 |
| EQUIPMENT | 13 |

## Edges by entity type pair

| Type pair | Edges |
| --- | --- |
| PROCESS — PROCESS | 31 |
| PROCESS — PROCESS PARAMETER | 23 |
| MICROSTRUCTURE — PHASE | 20 |
| PERSON — PERSON | 19 |
| MECHANICAL PROPERTY — PROCESS | 16 |
| ORGANIZATION — PERSON | 14 |
| MICROSTRUCTURE — PROCESS | 14 |
| PHENOMENON — PROCESS | 12 |
| MECHANICAL PROPERTY — PHENOMENON | 12 |
| PROCESS — STEEL GRADE | 11 |
| PHASE — PROCESS | 11 |
| EQUIPMENT — PROCESS | 11 |
| ALLOYING ELEMENT — MATERIAL | 11 |
| PHENOMENON — PHENOMENON | 11 |
| ALLOYING ELEMENT — STEEL GRADE | 10 |
| PROCESS PARAMETER — PROCESS PARAMETER | 9 |
| PHASE — PHENOMENON | 9 |
| PHASE — PHASE | 8 |
| MECHANICAL PROPERTY — PHASE | 8 |
| ORGANIZATION — PROCESS | 7 |
| MECHANICAL PROPERTY — MECHANICAL PROPERTY | 7 |
| MICROSTRUCTURE — PHENOMENON | 7 |
| PHENOMENON — PROCESS PARAMETER | 7 |
| MICROSTRUCTURE — PROCESS PARAMETER | 6 |
| MATERIAL — PROCESS | 6 |
| MECHANICAL PROPERTY — MICROSTRUCTURE | 6 |
| MECHANICAL PROPERTY — PROCESS PARAMETER | 6 |
| EQUIPMENT — MATERIAL | 5 |
| PERSON — PROCESS | 5 |
| MICROSTRUCTURE — MICROSTRUCTURE | 5 |
| EQUIPMENT — PROCESS PARAMETER | 4 |
| ALLOYING ELEMENT — PHENOMENON | 4 |
| ALLOYING ELEMENT — PROCESS | 3 |
| MATERIAL — STEEL GRADE | 3 |
| MATERIAL — PROCESS PARAMETER | 3 |
| MATERIAL — MICROSTRUCTURE | 3 |
| MATERIAL — PHASE | 3 |
| PHENOMENON — STEEL GRADE | 3 |
| MECHANICAL PROPERTY — STEEL GRADE | 3 |
| PERSON — PHASE | 3 |
| PROCESS PARAMETER — STEEL GRADE | 2 |
| EQUIPMENT — MECHANICAL PROPERTY | 2 |
| MATERIAL — MATERIAL | 2 |
| PHASE — STEEL GRADE | 2 |
| ALLOYING ELEMENT — MECHANICAL PROPERTY | 2 |
| MATERIAL — PHENOMENON | 2 |
| PHASE — PROCESS PARAMETER | 2 |
| PERSON — PHENOMENON | 2 |
| MATERIAL — MECHANICAL PROPERTY | 2 |
| MECHANICAL PROPERTY — ORGANIZATION | 2 |
| ORGANIZATION — ORGANIZATION | 2 |
| EQUIPMENT — EQUIPMENT | 1 |
| EQUIPMENT — PHENOMENON | 1 |
| MATERIAL — PERSON | 1 |
| PERSON — STEEL GRADE | 1 |
| STEEL GRADE — STEEL GRADE | 1 |
| PERSON — PROCESS PARAMETER | 1 |
| MICROSTRUCTURE — STEEL GRADE | 1 |

## Source book of entities

An entity is attributed to a book by the text units it was extracted from; `both` means
the same entity name was extracted from text units of both books and merged by GraphRAG.

| Source | Nodes |
| --- | --- |
| tanaka1981 | 248 |
| stat3 | 92 |

### Edges by source pair

| Source pair | Edges |
| --- | --- |
| tanaka1981 — tanaka1981 | 307 |
| stat3 — stat3 | 81 |

### Components by source mix

| Sources in component | Components |
| --- | --- |
| stat3 | 15 |
| tanaka1981 | 13 |

### Largest component by source

| Source | Nodes |
| --- | --- |
| tanaka1981 | 217 |

## Top 15 entities by degree

| # | Entity | Type | Source | Degree |
| --- | --- | --- | --- | --- |
| 1 | CONTROLLED ROLLING | PROCESS | tanaka1981 | 52 |
| 2 | RECRYSTALLIZATION | PHENOMENON | tanaka1981 | 20 |
| 3 | КП | PROCESS | stat3 | 17 |
| 4 | CONTROLLED-ROLLED STEEL | MATERIAL | tanaka1981 | 17 |
| 5 | GRAIN SIZE | MICROSTRUCTURE | tanaka1981 | 12 |
| 6 | DEFORMATION | PHENOMENON | tanaka1981 | 11 |
| 7 | YIELD STRENGTH | MECHANICAL PROPERTY | tanaka1981 | 11 |
| 8 | AUSTENITE | PHASE | tanaka1981 | 10 |
| 9 | MECHANICAL PROPERTIES | MECHANICAL PROPERTY | tanaka1981 | 10 |
| 10 | T_{RS} | PHASE | tanaka1981 | 10 |
| 11 | STEEL | MATERIAL | tanaka1981 | 9 |
| 12 | SEPARATIONS | PHENOMENON | tanaka1981 | 9 |
| 13 | T. TANAKA | PERSON | tanaka1981 | 8 |
| 14 | ALPHA | PHASE | tanaka1981 | 8 |
| 15 | GRAIN REFINEMENT | PROCESS | tanaka1981 | 8 |
