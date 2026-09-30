## Stage 4: tokenization

| metric | raw:tanaka1981 | cleaned:tanaka1981 | normalized:tanaka1981 | raw:stat3 | cleaned:stat3 | normalized:stat3 | raw:total | cleaned:total | normalized:total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| words | 12199 | 11932 | 12155 | 2138 | 2028 | 2288 | 14337 | 13960 | 14443 |
| words_inserted_by_normalization | 0 | 0 | 109 | 0 | 0 | 194 | 0 | 0 | 303 |
| unique_words | 1544 | 1512 | 1462 | 876 | 750 | 668 | 2416 | 2258 | 2086 |
| oov_words | 101 | 98 | 96 | 98 | 46 | 42 | 199 | 144 | 138 |
| oov_rate | 0.0083 | 0.0082 | 0.0079 | 0.0458 | 0.0227 | 0.0184 | 0.0139 | 0.0103 | 0.0096 |
| oov_rate_original_words | 0.0083 | 0.0082 | 0.008 | 0.0458 | 0.0227 | 0.0201 | 0.0139 | 0.0103 | 0.0098 |
| unique_oov | 41 | 38 | 38 | 84 | 34 | 26 | 125 | 72 | 64 |
| oov_examples | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, melloy | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, melloy | subgrain, bainitic, unrecrystallized, equiaxed, kozasu, melloy, hardenability, speciman | малоперлитных, рольганге, среднемассовой, трехстадийной, межкритическом, дительность, транспорти, ровка | малоперлитных, рольганге, среднемассовой, трехстадийной, межкритическом, клетями, нитридообразующие, непрерывнолитых | малоперлитная, среднемассовая, рольганг, подстуживание, непрерывнолитый, низколегированная, трехстадийная, межкритический | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, малоперлитных | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, малоперлитных, kozasu | subgrain, bainitic, unrecrystallized, equiaxed, малоперлитная, kozasu, melloy, среднемассовая |
| o200k_tokens | 22336 | 19072 | 18930 | 6887 | 6218 | 5995 | 29223 | 25290 | 24925 |
| o200k_tokens_per_word_whole_text | 1.831 | 1.598 | 1.557 | 3.221 | 3.066 | 2.62 | 2.038 | 1.812 | 1.726 |
| o200k_word_fertility | 1.183 | 1.183 | 1.206 | 1.78 | 1.798 | 1.7 | 1.272 | 1.273 | 1.284 |
| bge_tokens | 27175 | 23582 | 22756 | 7108 | 6549 | 6173 | 34283 | 30131 | 28929 |
| bge_tokens_per_word_whole_text | 2.228 | 1.976 | 1.872 | 3.325 | 3.229 | 2.698 | 2.391 | 2.158 | 2.003 |
| bge_word_fertility | 1.49 | 1.489 | 1.497 | 1.764 | 1.779 | 1.707 | 1.531 | 1.531 | 1.53 |
| bge_unk_tokens | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| formulas | 412 | 412 | 182 | 68 | 68 | 24 | 480 | 480 | 206 |
| formulas_converted_to_text | 0 | 0 | 230 | 0 | 0 | 44 | 0 | 0 | 274 |
| formulas_balanced_share | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| unmatched_math_delimiters | 7 | 0 | 0 | 1 | 0 | 0 | 8 | 0 | 0 |
