from enum import StrEnum

from pydantic import BaseModel


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    FORMULA = "formula"
    TABLE = "table"
    IMAGE = "image"


class Block(BaseModel):
    kind: BlockKind
    text: str
    start: int
    end: int


class Chunk(BaseModel):
    chunk_id: str
    book: str
    lang: str
    section: str
    text: str
    n_tokens: int
    start: int
    end: int


class AuditEntry(BaseModel):
    chunk_id: str
    rule: str
    before: str
    after: str


class TextStats(BaseModel):
    count: int
    mean: float
    median: float
    min: int
    max: int
    std: float


class ChunkingMetrics(BaseModel):
    n_chunks: int
    tokens: TextStats
    end_mid_sentence_share: float
    start_mid_sentence_share: float
    boundaries: int
    boundaries_cutting_sentence_share: float
    formulas_total: int
    formulas_broken: int
    formulas_broken_share: float
    formulas_not_whole_anywhere: int
    tables_total: int
    tables_broken: int
    tables_not_whole_anywhere: int
    over_graphrag_limit: int


class CleaningMetrics(BaseModel):
    characters: int
    numbers: int
    numbers_removed_by_deletion: int
    numbers_expected: int
    numbers_lost: int
    numbers_preserved_share: float
    formulas: int
    formulas_removed_by_deletion: int
    formulas_preserved_share: float
    tables: int
    table_cells: int
    table_cells_preserved_share: float
    table_cells_modified: int
    noise_characters: int
    noise_share: float
    service_marks: int
    pii_hits: int
    broken_paragraphs: int
    hyphenated_breaks: int
    ocr_corrections: int


class NormalizationMetrics(BaseModel):
    words: int
    words_changed: int | None
    words_changed_share: float | None
    annotation_events: int
    distinct_terms_canonicalized: int
    glossary_terms_found: int
    glossary_terms_with_english: int
    bridging_terms: int | None
    unit_expressions: int
    unit_expressions_canonical: int
    unit_canonical_share: float | None
    temperatures: int
    temperatures_canonical_share: float | None
    temperatures_in_range_share: float | None
    dates: int
    dates_valid_share: float | None
    lemmatized_words: int | None = None
    lemmatized_share: float | None = None
    full_lemmatization_share: float | None = None


class TokenizationMetrics(BaseModel):
    words: int
    words_inserted_by_normalization: int
    unique_words: int
    oov_words: int
    oov_rate: float
    oov_rate_original_words: float
    unique_oov: int
    oov_examples: list[str]
    o200k_tokens: int
    o200k_tokens_per_word_whole_text: float
    o200k_word_fertility: float
    bge_tokens: int
    bge_tokens_per_word_whole_text: float
    bge_word_fertility: float
    bge_unk_tokens: int
    formulas: int
    formulas_converted_to_text: int
    formulas_balanced_share: float
    unmatched_math_delimiters: int


class VectorizationMetrics(BaseModel):
    documents: int
    tfidf_windows: int
    tfidf_vocabulary: int
    tfidf_sparsity: float
    tfidf_mean_nonzero_per_doc: float
    dense_dimensions: int
    dense_mean_pairwise_cosine: float
    cross_language_mean_cosine: float | None
    cross_language_best_match_cosine: float | None


class GoldPageMetrics(BaseModel):
    gold_characters: int
    raw_cer: float
    clean_cer: float
    raw_wer: float
    clean_wer: float
    blocks: int
    blocks_should_delete: int
    blocks_deleted: int
    blocks_deleted_correctly: int
    deletion_precision: float
    deletion_recall: float
    deletion_f1: float
    units_expected: int
    units_correct: int
    unit_accuracy: float | None


class RetrievalMetrics(BaseModel):
    queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_10: float
