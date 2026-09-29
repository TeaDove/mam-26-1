## Stage 5: vectorization

| metric | dirty_units:tanaka1981 | cleaned:tanaka1981 | normalized:tanaka1981 | dirty_units:stat3 | cleaned:stat3 | normalized:stat3 | dirty_units:total | cleaned:total | normalized:total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| documents | 26 | 36 | 36 | 7 | 9 | 9 | 33 | 45 | 45 |
| tfidf_windows | 47 | 40 | 41 | 8 | 7 | 8 | 55 | 47 | 49 |
| tfidf_vocabulary | 1815 | 1512 | 1533 | 969 | 750 | 807 | 2777 | 2258 | 2296 |
| tfidf_sparsity | 0.9188 | 0.902 | 0.902 | 0.8075 | 0.7648 | 0.7756 | 0.9449 | 0.9325 | 0.9324 |
| tfidf_mean_nonzero_per_doc | 147.3 | 148.2 | 150.2 | 186.5 | 176.4 | 181.1 | 153 | 152.4 | 155.3 |
| dense_dimensions | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| dense_mean_pairwise_cosine | 0.6882 | 0.6703 | 0.6762 | 0.7704 | 0.7362 | 0.7637 | 0.6503 | 0.6259 | 0.6466 |
| cross_language_mean_cosine | — | — | — | — | — | — | 0.5686 | 0.5271 | 0.5762 |
| cross_language_best_match_cosine | — | — | — | — | — | — | 0.6251 | 0.6162 | 0.6808 |
