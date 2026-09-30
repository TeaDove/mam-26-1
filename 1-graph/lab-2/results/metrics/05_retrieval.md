## Stage 5: retrieval quality (bge-m3), chunking × preprocessing

| metric | dirty_windows:all | dirty_structural:all | clean_windows:all | clean_structural:all | dirty_windows:same_language | dirty_structural:same_language | clean_windows:same_language | clean_structural:same_language | dirty_windows:cross_language | dirty_structural:cross_language | clean_windows:cross_language | clean_structural:cross_language | dirty_windows:classmate_set | dirty_structural:classmate_set | clean_windows:classmate_set | clean_structural:classmate_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| queries | 26 | 26 | 26 | 26 | 14 | 14 | 14 | 14 | 12 | 12 | 12 | 12 | 10 | 10 | 10 | 10 |
| hit_at_1 | 0.3462 | 0.6154 | 0.4615 | 0.6923 | 0.4286 | 0.7143 | 0.5 | 0.7857 | 0.25 | 0.5 | 0.4167 | 0.5833 | 0.3 | 0.4 | 0.3 | 0.6 |
| hit_at_3 | 0.8077 | 0.6923 | 0.8077 | 0.8077 | 0.8571 | 0.7857 | 0.8571 | 0.8571 | 0.75 | 0.5833 | 0.75 | 0.75 | 0.7 | 0.4 | 0.7 | 0.7 |
| hit_at_5 | 0.8846 | 0.7692 | 1 | 0.9231 | 0.9286 | 0.8571 | 1 | 1 | 0.8333 | 0.6667 | 1 | 0.8333 | 0.8 | 0.5 | 1 | 0.8 |
| mrr | 0.5889 | 0.6988 | 0.6724 | 0.7908 | 0.6458 | 0.7768 | 0.6988 | 0.8571 | 0.5225 | 0.6077 | 0.6417 | 0.7133 | 0.5436 | 0.496 | 0.575 | 0.706 |
| ndcg_at_10 | 0.5805 | 0.6566 | 0.6748 | 0.725 | 0.6461 | 0.7611 | 0.72 | 0.8117 | 0.504 | 0.5347 | 0.622 | 0.624 | 0.5617 | 0.5091 | 0.6065 | 0.6342 |
