from .nodes import (
    retrieve,
    generate,
    grade_documents,
    web_search,
    LGBTQ_handler,
    answer_rag,
    reject_question,
    no_documents,
    no_answer,
    web_decision,
    gpt_decision,
    llm_fallback,
    retrieve_about_bot
)
from .edges import rag_router_edge, grade_generation_v_documents_and_question, gpt_allower, web_allower, decide_to_generate
from .state import GraphState
from .validator import TopicValidator
import asyncio
from httpx import AsyncClient

# Compile RAG graph
from langgraph.graph import END, StateGraph, START


rag_workflow = StateGraph(GraphState)

# Define nodes
rag_workflow.add_node("retrieve", retrieve)
rag_workflow.add_node("retrieve_about_bot", retrieve_about_bot)
rag_workflow.add_node("web_search", web_search)
rag_workflow.add_node("LGBTQ_handler", LGBTQ_handler)
rag_workflow.add_node("reject_question", reject_question)
rag_workflow.add_node("llm_fallback", llm_fallback)
rag_workflow.add_node("generate", generate)
rag_workflow.add_node("answer_rag", answer_rag)
rag_workflow.add_node("no_answer", no_answer)
rag_workflow.add_node("no_documents", no_documents)
rag_workflow.add_node("grade_documents", grade_documents)
rag_workflow.add_node('web_decision', web_decision)
rag_workflow.add_node("gpt_decision", gpt_decision)


rag_workflow.add_conditional_edges(
    START,
    rag_router_edge,
    {'vectorstore': 'retrieve',
     "bot_vectorstore": 'retrieve_about_bot',
     'web_search': "web_decision",
     "LGBTQ+_related": "LGBTQ_handler",
     "GPT": "gpt_decision",
     },
)

rag_workflow.add_edge("retrieve", "grade_documents")
rag_workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "No relevant documents": "no_documents",
    },
)

rag_workflow.add_edge('retrieve_about_bot', "generate")

rag_workflow.add_conditional_edges(
    "web_decision",
    web_allower,
    {
        "allowed": "web_search",
        "rejected": "reject_question"
    },
)

rag_workflow.add_edge("web_search", "generate")

rag_workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        # Because of issues we decided to ignore Hallucinations: answer_rag
        "not supported": "answer_rag",
        "not useful": "no_answer",  # Fails to answer question: uses no_answer node
        "useful": "answer_rag",
    },
)

rag_workflow.add_conditional_edges(
    "gpt_decision",
    gpt_allower,
    {
        "allowed": "llm_fallback",
        "rejected": "reject_question"
    }
)

rag_workflow.add_edge("LGBTQ_handler", END)
rag_workflow.add_edge("no_answer", END)
rag_workflow.add_edge("answer_rag", END)
rag_workflow.add_edge("reject_question", END)
rag_workflow.add_edge("llm_fallback", END)


rag_app = rag_workflow.compile()

if __name__ == "__main__":
    try:
        async def run_queries():
            questions = [
                "Can you book me an appointment",
                # "What is the founder of Encounter Jesus ministries international"
                # "Who are you"
                # "What is Apostle Jerry's Eze's minstry?",
                # "Who is Jesus Christ?"
                "How can i book an appointment with Apostle Uche Raymond"
            ]
            validator = TopicValidator()
            for question in questions:
                try:
                    answer = await rag_app.ainvoke(
                        {"question": question, 'validators': {
                            'validator': validator}}
                    )
                    print(answer['generation'])
                except BaseException as e:
                    print(f"Error for '{question}': {e}")

        asyncio.run(run_queries())
    except Exception as e:
        print(f"Unexpected error: {e}")
