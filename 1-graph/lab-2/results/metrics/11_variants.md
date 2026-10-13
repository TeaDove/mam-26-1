## Chunk overlap vs baseline (clean graph)

| metric | baseline | rerun | overlap | noise | delta | clear |
| --- | --- | --- | --- | --- | --- | --- |
| text units | 36 | 36 | 36 | 0 | +0 |  |
| input tokens | 24893 | 24893 | 28038 | 0 | +3145 | yes |
| vertices per 1k tokens | 17.03 | 15.63 | 12.34 | 1.4 | -4.69 | yes |
| vertices | 424 | 389 | 346 | 35 | -78 | yes |
| edges | 437 | 433 | 455 | 4 | +18 |  |
| connected components | 84 | 67 | 40 | 17 | -44 | yes |
| largest component share | 0.6604 | 0.6272 | 0.7717 | 0.0332 | +0.1113 | yes |
| isolated vertices | 64 | 49 | 30 | 15 | -34 | yes |
| RU vertices in largest component | 21 | 16 | 49 | 5 | +28 | yes |
| cycles per 100 vertices | 22.9 | 28.5 | 43.1 | 5.6 | +20.2 | yes |
| cross-book cycles | 71 | 74 | 128 | 3 | +57 | yes |
| completeness (semantic) | 0.6556 | 0.6722 | 0.5944 | 0.0166 | -0.0612 | yes |
| domain coverage | 0.925 | 0.95 | 0.95 | 0.025 | +0.025 |  |
| connected concept pairs | 9 | 8 | 9 | 1 | +0 |  |
| RU book share in BFS-2 | 0.195 | 0.1512 | 0.2972 | 0.0438 | +0.1022 | yes |
| noise share | 0.0249 | 0.0368 | 0.038 | 0.0119 | +0.0131 |  |
| formula vertices | 23 | 9 | 4 | 14 | -19 |  |
| translation duplicates | 2 | 3 | 1 | 1 | -1 |  |
| same-language duplicates | 11 | 6 | 4 | 5 | -7 |  |
| word-form duplicate vertices | 19 | 14 | 8 | 5 | -11 | yes |
| formula window share | 0.112 | 0.084 | 0.036 | 0.028 | -0.076 | yes |
| vertices with contradictions | 0.3083 | 0.2833 | 0.3083 | 0.0334 | +0 |  |
