from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from typing import Dict
from .book_app import BookAppointment
from app.db_logic.database import engine
from sqlalchemy.ext.asyncio import AsyncSession
from .acc_app import AcceptAppointment
from pydantic import BaseModel, Field
from typing import Literal, List
from .read_app import ReadAppointments
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
from .app_reminder import setup_scheduler
from .delete_app import DeleteAppointments


load_dotenv()

# STATE


class GraphState(TypedDict):
    user_request: str
    response: str
    user_name: str
    user_phone_number: str
    scheduler: List


# Nodes
async def book_appointment(state: GraphState):
    print("ROUTING TO BOOK APPOINTMENT")
    user_request = state['user_request']
    book_app = BookAppointment(user_request)
    user_name = state['user_name']
    user_phone_number = state['user_phone_number']

    response = await book_app.book_appointment(user_name, user_phone_number)
    return {'response': response}


async def accept_appointment(state: GraphState):
    print("ROUTING TO ACCEPT APPOINTMENT")
    user_request = state['user_request']
    user_phone_number = state['user_phone_number']
    scheduler = state['scheduler']
    async with AsyncSession(engine) as session:
        acc_app = AcceptAppointment(
            user_input=user_request, session=session, scheduler=scheduler[0])
        response = await acc_app.approve_appointment(user_phone_number)
    return {'response': response}


async def read_appointments(state: GraphState):
    print("ROUTING TO READ APPOINTMENTS")
    user_request = state['user_request']
    user_phone_number = state['user_phone_number']
    async with AsyncSession(engine) as session:
        read_app = ReadAppointments(user_request, session)
        response = await read_app.get_appointments_by_criteria(user_phone_number)
    return {'response': response}


async def delete_appointments(state: GraphState):
    print("Routing to delete appointements")
    user_request = state['user_request']
    user_phone_number = state['user_phone_number']
    scheduler = state['scheduler']
    async with AsyncSession(engine) as session:
        del_app = DeleteAppointments(user_request, session, scheduler[0])
        response = await del_app.delete_appointments(user_phone_number)
    return {'response': response}


def none(state: GraphState):
    print("ROUTING TO NONE")
    user_name = state['user_name']
    user_number = state['user_phone_number']
    allowed_numbers = ['2349094540644', '2348032235209',
                       '2349035476815', '2347052052592', '2348130332520']
    if user_number not in allowed_numbers:
        response = f"Hello {user_name} it seems you tried to perform an action you are not authorized for concerning appointments. You are only allowed to book appointments"
    else:
        response = f"Hello {user_name} only these actions concerning appointments can be performed, booking, accepting and reading"
    return {"response": response}

# Edges


async def route_app(state: GraphState):
    user_request = state['user_request']

    class RouteQuery(BaseModel):
        """Route a user query to the most relevant event."""

        datasource: Literal["book_appointment", "accept_appointment", "read_appointments", "delete_appointment", "None"] = Field(
            ...,
            description="Given a user statement route it book_appointment, accept_appointment, read_appointments, None",
        )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = """You are a smart router for an appointment management system. Analyze user input to determine their primary intent, focusing on meaning, not just keywords. Route to one of:
1. book_appointment: User wants to schedule a new appointment (e.g., "book", "schedule", "need a slot"). Examples: "Book 12th October", "Meeting tomorrow", "Fit me in Friday".
2. accept_appointment: User confirms an existing appointment, often with user_id/date (e.g., "accept", "confirm", "that works"). Examples: "Accept user_id 2, 12th October", "Confirm 12pm", "Works for me".
3. read_appointments: User wants to view appointments, often with time references (e.g., "read", "check", "show"). Examples: "Read appointments tomorrow", "Show next week", "List schedule".
4. delete_appointment: User wants to cancel/delete an appointment, often with ID/date (e.g., "cancel", "delete", "remove"). Examples: "Cancel 123", "Delete tomorrow", "Remove ID 456".
5. None: Unrelated or vague input. Examples: "Weather?", "Joke", "appointment".

**Rules**:
Prioritize first matching intent (e.g., "Book and cancel" → book_appointment).
Handle typos (e.g., "appoitment" → "appointment", "cancell" → "cancel").
Interpret vague inputs (e.g., "Slot ASAP" → book_appointment, "Drop meeting" → delete_appointment).
Case-insensitive.
Return one of: book_appointment, accept_appointment, read_appointments, delete_appointment, None."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router

    response = await question_router.ainvoke({"question": user_request})

    # response = response.datasource
    print(f"ROUTING TO {response.datasource}")

    return response.datasource

app_workflow = StateGraph(GraphState)
app_workflow.add_node('book_app', book_appointment)
app_workflow.add_node('acc_app', accept_appointment)
app_workflow.add_node('read_app', read_appointments)
app_workflow.add_node('del_app', delete_appointments)
app_workflow.add_node("none", none)

app_workflow.add_conditional_edges(
    START,
    route_app,
    {"book_appointment": 'book_app',
     "accept_appointment": 'acc_app',
     "read_appointments": "read_app",
     "delete_appointment": 'del_app',
     "None": "none"},
)

app_workflow.add_edge('book_app', END)
app_workflow.add_edge('acc_app', END)
app_workflow.add_edge('read_app', END)
app_workflow.add_edge('del_app', END)
app_workflow.add_edge("none", END)

app_workflow = app_workflow.compile()

if __name__ == "__main__":
    async def run_queries():
        questions = [
            # "Who is the president of france",

            "Book me an appointment for 10th June 2025 at 12:34",
            # "Accept this appointment user_id is 1, date is 11th May 2025, time is 22:05"
            "Read my appointments for today",
            # "Cancel appointment number 55"
        ]
        scheduler = setup_scheduler()
        scheduler.start()
        for question in questions:
            try:
                answer = await app_workflow.ainvoke(
                    {'user_request': question, "user_name": "Akachi",
                        "user_phone_number": "2349094540644", 'scheduler': [scheduler]}
                )
                print(answer['response'])
            except BaseException as e:
                print(f"Error for '{question}': {e}")
        scheduler.shutdown()

# Run all queries in a single event loop
    try:
        asyncio.run(run_queries())
    except Exception as e:
        print(f"Unexpected error: {e}")
