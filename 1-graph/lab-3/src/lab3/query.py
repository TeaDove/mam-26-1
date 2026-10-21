from lab3.llm import LlmClient
from lab3.models import QueryEntity, QueryGraph

ENTITY_TYPES = (
    "material, steel grade, alloying element, process, process parameter, mechanical property, microstructure, "
    "phase, equipment, phenomenon, person, organization"
)

QUERY_SYSTEM = (
    "You turn a user question about controlled rolling of steel into a small knowledge graph, the same way the "
    "knowledge base was built. Extract the domain entities the question is about and the relations between them "
    f"that the question asks for. Entity types: {ENTITY_TYPES}. Entity name: the canonical English term in upper "
    "case (translate Russian terms); keep abbreviations, designations and proper names as written (e.g. УОВТ, "
    "КП+УО+КП, USINOR, NB). Entity description: one short sentence on what the question asks about this entity. "
    "Relations connect two extracted entities; description: what link between them the question asks about. "
    'Answer JSON: {"entities": [{"name": "...", "type": "...", "description": "..."}], '
    '"relations": [{"source": "...", "target": "...", "description": "..."}]}. 1–5 entities.'
)


def extract_query_graph(client: LlmClient, question: str) -> QueryGraph:
    graph = client.complete(QUERY_SYSTEM, f"QUESTION: {question}", QueryGraph)
    names = {entity.name for entity in graph.entities}
    relations = [r for r in graph.relations if r.source in names and r.target in names and r.source != r.target]
    entities = graph.entities or [QueryEntity(name=question, description=question)]
    return QueryGraph(entities=entities, relations=relations)
