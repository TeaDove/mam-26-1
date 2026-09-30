## Stage 4: tokenization

| metric | raw:tanaka1981 | cleaned:tanaka1981 | normalized:tanaka1981 | raw:stat3 | cleaned:stat3 | normalized:stat3 | raw:total | cleaned:total | normalized:total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| words | 14077 | 11932 | 12191 | 2309 | 2028 | 2355 | 16386 | 13960 | 14546 |
| words_inserted_by_normalization | 0 | 0 | 145 | 0 | 0 | 261 | 0 | 0 | 406 |
| unique_words | 1815 | 1512 | 1533 | 969 | 750 | 807 | 2777 | 2258 | 2296 |
| oov_words | 245 | 98 | 98 | 117 | 46 | 46 | 362 | 144 | 144 |
| oov_rate | 0.0174 | 0.0082 | 0.008 | 0.0507 | 0.0227 | 0.0195 | 0.0221 | 0.0103 | 0.0099 |
| oov_rate_original_words | 0.0174 | 0.0082 | 0.0081 | 0.0507 | 0.0227 | 0.022 | 0.0221 | 0.0103 | 0.0102 |
| unique_oov | 107 | 38 | 39 | 100 | 34 | 34 | 207 | 72 | 73 |
| oov_examples | tetsu-to-hagané, kozasu, subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, melloy | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, melloy | малоперлитных, рольганге, среднемассовой, ичм, трехстадийной, межкритическом, непрерывнолитого, дительность | малоперлитных, рольганге, среднемассовой, трехстадийной, межкритическом, клетями, нитридообразующие, непрерывнолитых | малоперлитных, рольганге, среднемассовой, трехстадийной, межкритическом, клетями, нитридообразующие, непрерывнолитых | tetsu-to-hagané, kozasu, subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, малоперлитных, kozasu | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, малоперлитных, kozasu |
| o200k_tokens | 28155 | 19066 | 19100 | 7410 | 6217 | 6267 | 35565 | 25283 | 25367 |
| o200k_tokens_per_word_whole_text | 2 | 1.598 | 1.567 | 3.209 | 3.066 | 2.661 | 2.17 | 1.811 | 1.744 |
| o200k_word_fertility | 1.214 | 1.183 | 1.212 | 1.797 | 1.798 | 1.742 | 1.296 | 1.273 | 1.298 |
| bge_tokens | 32939 | 23582 | 23149 | 7595 | 6549 | 6527 | 40534 | 30131 | 29676 |
| bge_tokens_per_word_whole_text | 2.34 | 1.976 | 1.899 | 3.289 | 3.229 | 2.772 | 2.474 | 2.158 | 2.04 |
| bge_word_fertility | 1.503 | 1.489 | 1.521 | 1.778 | 1.779 | 1.786 | 1.542 | 1.531 | 1.564 |
| bge_unk_tokens | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| formulas | 412 | 412 | 182 | 68 | 68 | 24 | 480 | 480 | 206 |
| formulas_converted_to_text | 0 | 0 | 230 | 0 | 0 | 44 | 0 | 0 | 274 |
| formulas_balanced_share | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| unmatched_math_delimiters | 7 | 0 | 0 | 1 | 0 | 0 | 8 | 0 | 0 |
