from typing_extensions import TypedDict
from typing import Dict, Union, List


class GraphState(TypedDict):
    """state on main graph"""
    output_format:  str
    response: Union[str, Dict, List]
    p_and_c_validators: Dict
    rag_validator: List
    user_name: str
    user_phone_number: str
    user_request: str
    scheduler: List
    answered: bool
