## Stage 4: tokenization

| metric | raw:tanaka1981 | cleaned:tanaka1981 | normalized:tanaka1981 | raw:stat3 | cleaned:stat3 | normalized:stat3 | raw:total | cleaned:total | normalized:total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| words | 12199 | 11932 | 12153 | 2138 | 2028 | 2290 | 14337 | 13960 | 14443 |
| words_inserted_by_normalization | 0 | 0 | 102 | 0 | 0 | 196 | 0 | 0 | 298 |
| unique_words | 1544 | 1512 | 1493 | 876 | 750 | 706 | 2416 | 2258 | 2155 |
| oov_words | 101 | 98 | 98 | 98 | 46 | 44 | 199 | 144 | 142 |
| oov_rate | 0.0083 | 0.0082 | 0.0081 | 0.0458 | 0.0227 | 0.0192 | 0.0139 | 0.0103 | 0.0098 |
| oov_rate_original_words | 0.0083 | 0.0082 | 0.0081 | 0.0458 | 0.0227 | 0.021 | 0.0139 | 0.0103 | 0.01 |
| unique_oov | 41 | 38 | 38 | 84 | 34 | 29 | 125 | 72 | 67 |
| oov_examples | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, melloy | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, melloy | subgrain, bainitic, unrecrystallized, equiaxed, nucleates, kozasu, melloy, hardenability | малоперлитных, рольганге, среднемассовой, трехстадийной, межкритическом, дительность, транспорти, ровка | малоперлитных, рольганге, среднемассовой, трехстадийной, межкритическом, клетями, нитридообразующие, непрерывнолитых | малоперлитная, среднемассовая, рольганг, подстуживание, низколегированная, трехстадийной, межкритический, нитридообразующие | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, kozasu, малоперлитных | subgrains, bainitic, unrecrystallized, equiaxed, subgrain, nucleates, малоперлитных, kozasu | subgrain, bainitic, unrecrystallized, equiaxed, nucleates, малоперлитная, kozasu, melloy |
| o200k_tokens | 22336 | 19072 | 18896 | 6887 | 6218 | 5991 | 29223 | 25290 | 24887 |
| o200k_tokens_per_word_whole_text | 1.831 | 1.598 | 1.555 | 3.221 | 3.066 | 2.616 | 2.038 | 1.812 | 1.723 |
| o200k_word_fertility | 1.183 | 1.183 | 1.205 | 1.78 | 1.798 | 1.695 | 1.272 | 1.273 | 1.283 |
| bge_tokens | 27175 | 23582 | 22786 | 7108 | 6549 | 6210 | 34283 | 30131 | 28996 |
| bge_tokens_per_word_whole_text | 2.228 | 1.976 | 1.875 | 3.325 | 3.229 | 2.712 | 2.391 | 2.158 | 2.008 |
| bge_word_fertility | 1.49 | 1.489 | 1.501 | 1.764 | 1.779 | 1.721 | 1.531 | 1.531 | 1.536 |
| bge_unk_tokens | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| formulas | 412 | 412 | 182 | 68 | 68 | 24 | 480 | 480 | 206 |
| formulas_converted_to_text | 0 | 0 | 230 | 0 | 0 | 44 | 0 | 0 | 274 |
| formulas_balanced_share | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| unmatched_math_delimiters | 7 | 0 | 0 | 1 | 0 | 0 | 8 | 0 | 0 |
