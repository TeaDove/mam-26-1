from enum import StrEnum

from pydantic import BaseModel, Field


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
    terms_expected: int
    terms_correct: int
    term_accuracy: float | None


class RetrievalMetrics(BaseModel):
    queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_10: float


class GraphSummary(BaseModel):
    vertices: int
    edges: int
    input_tokens: int


class InputStats(BaseModel):
    text_units: int
    input_tokens: int
    vertices_per_1k_tokens: float
    noise_vertices_per_1k_tokens: float


class GraphStructure(BaseModel):
    nodes: int
    edges: int
    connected_components: int
    largest_component_share: float
    isolated_nodes: int
    bridges: int
    articulation_points: int
    cyclomatic_number: int
    degree_mean: float
    degree_max: int
    leaves: int
    nodes_ru: int
    nodes_both_languages: int
    edges_en_ru: int
    edges_touching_bilingual_nodes: int
    ru_nodes_in_largest_component: int
    nodes_by_type: dict[str, int]
    edges_by_type_pair: dict[str, int]


class Completeness(BaseModel):
    reference_terms: int
    covered_semantic: int
    covered_semantic_share: float


class Consistency(BaseModel):
    checked_vertices: int
    fully_defined_share: float
    contradictions_total: int
    vertices_with_contradictions_share: float


class Coreference(BaseModel):
    lemma_duplicate_groups: int
    lemma_duplicate_vertices: int
    embedding_duplicate_pairs: int
    pronoun_vertices_lifted: int
    lemma_examples: list[str]


class NoiseReport(BaseModel):
    vertices: int
    noise_vertices: int
    noise_share: float
    by_rule: dict[str, int]
    examples: list[str]


class Integrity(BaseModel):
    formulas: int
    formulas_evaluated: int
    formulas_in_vertex: int
    formulas_in_window: int
    formula_mean_window_share: float
    tables_evaluated: int
    tables_broken: int
    table_best_window_share: list[float]


class Coverage(BaseModel):
    concepts_total: int
    concepts_found: int
    domain_coverage: float
    vector_coverage: float


class TraversalSummary(BaseModel):
    seeds_resolved: int
    pairs_total: int
    pairs_resolved: int
    pairs_connected: int
    bfs1_mean_reached: float
    bfs2_mean_reached: float
    bfs3_mean_reached: float
    bfs2_mean_concepts: float
    bfs2_noise_share: float
    bfs2_ru_book_share: float
    dfs_mean_reached: float


class PathPair(BaseModel):
    source: str
    target: str
    path: list[str] | None = None


class Traversal(BaseModel):
    summary: TraversalSummary
    pairs: list[PathPair]


class GraphReport(BaseModel):
    source: InputStats = Field(alias="input")
    structure: GraphStructure
    completeness: Completeness
    consistency: Consistency
    coreference: Coreference
    noise: NoiseReport
    integrity: Integrity
    coverage: Coverage
    traversal: Traversal


class GraphComparison(BaseModel):
    dirty: GraphReport
    clean: GraphReport
