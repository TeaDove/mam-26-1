from pydantic import BaseModel


class Question(BaseModel):
    id: str
    lang: str
    book: str
    question: str
    evidence: list[str]
    key: str
    answer: str


class QuestionSet(BaseModel):
    questions: list[Question]


class QueryEntity(BaseModel):
    name: str
    type: str = ""
    description: str = ""


class QueryRelation(BaseModel):
    source: str
    target: str
    description: str = ""


class QueryGraph(BaseModel):
    entities: list[QueryEntity] = []
    relations: list[QueryRelation] = []


class Answer(BaseModel):
    answer: str


class Verdict(BaseModel):
    verdict: str
    comment: str = ""


class LinkMetrics(BaseModel):
    graph: str
    encoder: str
    seed: int
    vertices: int
    edges: int
    best_epoch: int
    cosine_test_auc: float
    val_auc: float
    test_auc: float
    test_ap: float


class QuestionResult(BaseModel):
    id: str
    graph: str
    retriever: str
    question: str
    reference: str
    answer: str
    verdict: str
    comment: str
    query_entities: list[str]
    seeds: list[str]
    context: list[str]
    source_units: list[str]
    relevant_vertices: int
    first_relevant_rank: int | None
    context_has_relevant: bool
    key_in_graph: bool
    key_in_context: bool


class RunMetrics(BaseModel):
    graph: str
    retriever: str
    questions: int
    with_relevant: int
    mrr: float
    hit_at_1: float
    hit_at_3: float
    hit_at_10: float
    context_recall: float
    key_in_graph_share: float
    key_in_context_share: float
    correct_share: float
    correct_or_partial_share: float


class RagReport(BaseModel):
    runs: list[RunMetrics]
    link_prediction: list[LinkMetrics]


class RetrievalMetrics(BaseModel):
    graph: str
    retriever: str
    questions: int
    with_relevant: int
    mrr: float
    hit_at_1: float
    hit_at_3: float
    hit_at_10: float
    context_recall: float
    key_in_context_share: float
