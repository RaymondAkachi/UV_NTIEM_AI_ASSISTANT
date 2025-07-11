from langgraph.graph import StateGraph, START, END
from .main_graph_state import GraphState
from .main_graph_n_and_e import watch_sermon_node, image_creation_node, rag_node, query_rewriter_node, p_and_c_node, app_node, q_router
from functools import lru_cache
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import time
import asyncio
import traceback


@lru_cache(maxsize=1)
def create_workflow_graph():
    main_graph_workflow = StateGraph(GraphState)
    main_graph_workflow.add_node("query_rewriter", query_rewriter_node)
    main_graph_workflow.add_node("watch_sermon", watch_sermon_node)
    main_graph_workflow.add_node("appointments", app_node)
    main_graph_workflow.add_node("image_creation", image_creation_node)
    main_graph_workflow.add_node("prayer_and_counselling", p_and_c_node)
    main_graph_workflow.add_node("RAG", rag_node)

    main_graph_workflow.add_edge(START, "query_rewriter")
    main_graph_workflow.add_conditional_edges(
        "query_rewriter",
        q_router,
        {"Appointment": 'appointments',
         "Answered": END,
         "Prayer or Counselling": 'prayer_and_counselling',
         "Watch Sermon": "watch_sermon",
         "Image Creation": "image_creation",
         "None": "RAG"},
    )

    main_graph_workflow.add_edge('appointments', END)
    main_graph_workflow.add_edge('prayer_and_counselling', END)
    main_graph_workflow.add_edge('watch_sermon', END)
    main_graph_workflow.add_edge("image_creation", END)
    main_graph_workflow.add_edge("RAG", END)

    return main_graph_workflow


if __name__ == "__main__":
    from p_and_c.prayer_embeddings import PrayerRelation
    from p_and_c.counselling_embeddings import CounsellingRelation
    from RAG.validator import TopicValidator
    from appointment_logic.app_reminder import setup_scheduler

    async def execute_main_graph():
        user_name = "Akachi"
        user_phone_number = '2349094540644'
        session_id = user_phone_number

        # Prepare the validators, scheduler, and graph builder
        rag_validator = [TopicValidator()]
        p_and_c_validators = {
            'validator': PrayerRelation(),
            "counselling_validator": CounsellingRelation()
        }
        scheduler = setup_scheduler()
        graph = create_workflow_graph().compile()
        for i in ["I need prayer my wife is sick", "Generate an image of a dancing gorilla", "Get me the sermon titled 'test_video2'"]:
            a = time.time()
            try:
                results = await graph.ainvoke(
                    {
                        'user_request': i,
                        'user_name': user_name,
                        'user_phone_number': user_phone_number,
                        'rag_validator': rag_validator,
                        'p_and_c_validators': p_and_c_validators,
                        'scheduler': [scheduler]
                    })
                b = time.time()
                print(b-a)
                print(results['output_format'], results['response'])
            except BaseException as e:
                print(f"Error occurred: {e}")

    asyncio.run(execute_main_graph())


# if __name__ == "__main__":
#     async def execute_main_graph():
#         try:
#             user_name = "Akachi1239978"
#             user_phone_number = '2349096760695'
#             session_id = user_phone_number

#             # Prepare the validators, scheduler, and graph builder
#             rag_validator = [TopicValidator()]
#             # p_and_c_validators = {
#             #     'validator': PrayerRelation(),
#             #     "counselling_validator": CounsellingRelation()
#             # }
#             p_and_c_validators = {'validator': "", "counselling_validator": ""}
#             scheduler = setup_scheduler()
#             graph_builder = create_workflow_graph()

#             # Prepare the configurable section of the config
#             configurable = {
#                 'thread_id': session_id
#             }

#             # Create a memory for short-term storage
#             async with AsyncSqliteSaver.from_conn_string("memory.db") as short_term_memory:
#                 graph = graph_builder.compile(checkpointer=short_term_memory)

#                 # Run the graph with proper inputs
#                 for user_request in [
#                     'Who is Apostle Micheal Orokpo',
#                         "Make me a picture of a Tiger"]:
#                     results = await graph.ainvoke(
#                         {
#                             'user_request': user_request,
#                             'user_name': user_name,
#                             'user_phone_number': user_phone_number,
#                             'rag_validator': rag_validator,
#                             'p_and_c_validators': p_and_c_validators,
#                             'scheduler': [scheduler]
#                         },
#                         # Pass configurable section here
#                         {"configurable": configurable}
#                     )
#                     print(results['output_format'], results['response'])

#         except BaseException as e:
#             print("Error occurred:\n%s" % traceback.format_exc())

#     # Call the main graph execution function
#     asyncio.run(execute_main_graph())
