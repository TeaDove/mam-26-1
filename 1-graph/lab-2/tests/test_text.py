from lab2.dto.models import BlockKind
from lab2.util.text import clean_cut, formulas, is_caption, is_continuation, parse_blocks


def test_parse_blocks_keeps_display_formula_whole() -> None:
    text = "Intro text here.\n\n$$\na = b\n\nc = d\n$$\n\nwhere a is x."
    blocks = parse_blocks(text)
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.FORMULA, BlockKind.PARAGRAPH]
    assert all(text[b.start : b.end].strip() == b.text for b in blocks)


def test_caption_heading_is_paragraph() -> None:
    blocks = parse_blocks("## 37 Structures in 0.06C steel")
    assert blocks[0].kind is BlockKind.PARAGRAPH
    assert is_caption(blocks[0])


def test_citations_are_not_formulas() -> None:
    assert formulas("Arrowsmith $^{1}$ showed $\\gamma$ grains $^{27,36}$") == ["$\\gamma$"]


def test_continuation_detection() -> None:
    assert is_continuation("greater than those of highly", "alloyed or heat-treated steels.")
    assert is_continuation("The reason for the im-", "Provement is clear.")
    assert is_continuation("shown in", "Fig. 23, where")
    assert not is_continuation("This is a sentence.", "another one")


def test_clean_cut_rejects_formula_start_and_accepts_sentence_end() -> None:
    blocks = parse_blocks("First sentence ends here properly.\n\nSecond paragraph starts.\n\n$$\nx\n$$")
    assert clean_cut(blocks[:1], blocks[1:])
    assert not clean_cut(blocks[:2], blocks[2:])
