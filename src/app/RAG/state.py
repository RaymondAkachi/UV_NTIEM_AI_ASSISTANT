from typing import List, Dict

from typing_extensions import TypedDict


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    validators: Dict
    workflow: str
    inner_source: str
