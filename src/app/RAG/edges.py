from .chains import rag_router, hallucination_grader, answer_grader
from .state import GraphState
# import asyncio
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


async def rag_router_edge(state: GraphState) -> Literal["vectorstore", "bot_vectorstore", "web_search", "LGBTQ+_related", "GPT"]:
    question = state['question']

    chain = rag_router()
    result = await chain.ainvoke({'question': question})
    print(f"---ROUTING QUESTION TO {result.datasource}---")
    return result.datasource


# chain = rag_router()
# res = asyncio.run(chain.ainvoke({'question': "Who is Apostle Uche Raymond"}))
# print(res)
def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "No relevant documents"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


async def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_chain = hallucination_grader()
    score = await hallucination_chain.ainvoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        answer_chain = answer_grader()
        score = await answer_chain.ainvoke(
            {"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


async def web_allower(state: GraphState):
    question = state['question']
    validator = state['validators']['validator']
    res = (await validator.explain_similarity(question))['decision'] == 'allowed'
    if res:
        return "allowed"
    else:
        return "rejected"


async def gpt_allower(state: GraphState):
    question = state['question']
    validator = state['validators']['validator']
    res = (await validator.explain_similarity(question))['decision'] == 'allowed'
    if res:
        return "allowed"
    else:
        return "rejected"
