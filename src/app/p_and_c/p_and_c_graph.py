from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from .p_and_c_embeddings import p_and_c_router, prayer_and_counselling
# from .prayer_embeddings import PrayerRelation
# from .counselling_embeddings import CounsellingRelation
from typing import Dict
# import asyncio


class GraphState(TypedDict):
    intent: str
    request: str
    response: str
    validators: Dict


# Nodes
# def prayer(state: GraphState):
async def prayer(state: GraphState):
    print('Routing to Prayer Node')
    validator = state['validators']['prayer_validator']
    request = state['request']
    response = await validator.return_help(request)
    return {'response': response, 'intent': 'prayer'}


async def counselling(state: GraphState):
    print("Routing to counselling node")
    validator = state['validators']['counselling_validator']
    request = state['request']
    response = await validator.return_help(request)
    return {'response': response, 'intent': "counselling"}


async def both_p_and_c(state: GraphState):
    print("Routing to P and C")
    request = state['request']
    validator = state['validators']['prayer_validator']
    counselling_validator = state['validators']['counselling_validator']
    response = await prayer_and_counselling(
        request, validator=validator, counselling_validator=counselling_validator)
    return {'response': response}

# Edges


async def router(state: GraphState):
    request = state['request']
    chain = p_and_c_router()
    intent = await chain.ainvoke({'question': request})
    return intent.datasource


help_workflow = StateGraph(GraphState)
help_workflow.add_node('prayer', prayer)
help_workflow.add_node('counselling', counselling)
help_workflow.add_node('both_p_and_c', both_p_and_c)

help_workflow.add_conditional_edges(
    START,
    router,
    {"PRAYER": 'prayer',
     "COUNSELLING": 'counselling',
     "BOTH": "both_p_and_c"},
)

help_workflow.add_edge('prayer', END)
help_workflow.add_edge('counselling', END)
help_workflow.add_edge('prayer', END)

help_workflow = help_workflow.compile()


# Run all queries in a single event loop
# if __name__ == "__main__":
#     try:
#         async def run_queries():
#             questions = [
#                 # "Who is the president of france",
#                 "I need prayer for healing of my husband?",
#                 # "I need counselling in am confused",
#                 "I need couselling for addiction",
#                 "I need prayer my cousin is sick and counselling for my marriage",
#                 # "I need prayer for my marriage"

#             ]
#             validator = PrayerRelation()
#             counselling_validator = CounsellingRelation()
#             for question in questions:
#                 try:
#                     answer = await help_workflow.ainvoke(
#                         {"request": question, 'validators': {
#                             'prayer_validator': validator, "counselling_validator": counselling_validator}}
#                     )
#                     print(answer['response'])
#                 except BaseException as e:
#                     print(f"Error for '{question}': {e}")
#         asyncio.run(run_queries())
#     except Exception as e:
#         print(f"Unexpected error: {e}")
