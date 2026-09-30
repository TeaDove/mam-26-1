# Gold pages

Hand-made reference ("gold") text for four PDF pages, used to score the cleaning and normalization stages.

| Page | Why this page |
| --- | --- |
| tanaka1981 p.186 | running header/footer, page number, citation markers, a word broken across paragraphs |
| tanaka1981 p.202 | display formulas with OCR noise, figure captions splitting a paragraph |
| stat3 p.262 | Russian hyphenation, figure legend, OCR glyph instead of a list number |
| stat3 p.267 | the only table: hyphenated header cells, OCR errors in subscripts |

Files per page:

- `*.raw.md` — the page segment of the raw MinerU markdown (bibliography removed), between page-number lines;
- `*.gold.md` — the page as it should read after cleaning: only content, verified against the PDF page image;
- `*.units.txt` — quantities as they should read after normalization, one per line;
- `*.terms.txt` — terms as they should read after normalization (canonical English term first, lemmatized original in brackets), one per line.

The gold text was written by the author from the PDF page images; an independent check by the team is pending.
