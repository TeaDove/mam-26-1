## RAG

| graph | retriever | MRR | Hit@1 | Hit@3 | Hit@10 | context recall | correct | correct+partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dirty | bge-m3 | 0.602 | 0.50 | 0.62 | 0.85 | 0.69 | 0.12 | 0.31 |
| dirty | gat | 0.345 | 0.27 | 0.27 | 0.62 | 0.73 | 0.12 | 0.19 |
| dirty | sage | 0.346 | 0.23 | 0.42 | 0.62 | 0.69 | 0.12 | 0.15 |
| clean | bge-m3 | 0.744 | 0.62 | 0.88 | 0.88 | 0.92 | 0.27 | 0.50 |
| clean | gat | 0.487 | 0.38 | 0.54 | 0.73 | 0.81 | 0.19 | 0.31 |
| clean | sage | 0.639 | 0.54 | 0.69 | 0.85 | 0.92 | 0.12 | 0.23 |

## Link prediction

| graph | encoder | best epoch | val ROC-AUC | test ROC-AUC | test AP |
| --- | --- | --- | --- | --- | --- |
| dirty | gat | 26 | 0.695 | 0.790 | 0.813 |
| dirty | sage | 68 | 0.521 | 0.635 | 0.635 |
| clean | gat | 1 | 0.699 | 0.562 | 0.573 |
| clean | sage | 28 | 0.592 | 0.547 | 0.548 |
