## Stage 5: retrieval quality (bge-m3), chunking × preprocessing

| metric | dirty_windows:all | dirty_structural:all | clean_windows:all | clean_structural:all | dirty_windows:same_language | dirty_structural:same_language | clean_windows:same_language | clean_structural:same_language | dirty_windows:cross_language | dirty_structural:cross_language | clean_windows:cross_language | clean_structural:cross_language | dirty_windows:classmate_set | dirty_structural:classmate_set | clean_windows:classmate_set | clean_structural:classmate_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| queries | 26 | 26 | 26 | 26 | 14 | 14 | 14 | 14 | 12 | 12 | 12 | 12 | 10 | 10 | 10 | 10 |
| hit_at_1 | 0.3462 | 0.6154 | 0.5 | 0.6538 | 0.4286 | 0.7143 | 0.5714 | 0.7143 | 0.25 | 0.5 | 0.4167 | 0.5833 | 0.3 | 0.4 | 0.4 | 0.6 |
| hit_at_3 | 0.8077 | 0.6923 | 0.8077 | 0.8462 | 0.8571 | 0.7857 | 0.8571 | 0.8571 | 0.75 | 0.5833 | 0.75 | 0.8333 | 0.7 | 0.4 | 0.7 | 0.7 |
| hit_at_5 | 0.8846 | 0.7692 | 0.9615 | 0.9231 | 0.9286 | 0.8571 | 1 | 1 | 0.8333 | 0.6667 | 0.9167 | 0.8333 | 0.8 | 0.5 | 0.9 | 0.8 |
| mrr | 0.5889 | 0.6988 | 0.6923 | 0.7738 | 0.6458 | 0.7768 | 0.7381 | 0.8214 | 0.5225 | 0.6077 | 0.6389 | 0.7183 | 0.5436 | 0.496 | 0.6167 | 0.7036 |
| ndcg_at_10 | 0.5805 | 0.6566 | 0.6692 | 0.7113 | 0.6461 | 0.7611 | 0.7264 | 0.7853 | 0.504 | 0.5347 | 0.6025 | 0.6251 | 0.5617 | 0.5091 | 0.6036 | 0.6328 |
