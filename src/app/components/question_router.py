# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time  # For in-script testing
import asyncio  # For in-script testing
from dotenv import load_dotenv

load_dotenv()


def is_question_checker(user_input: str):
    question_checker_prompt = """
        Determine if the input is a question, starting with words like "can", "what", "when", "where", "why", "how", "is", "are", "do", "does", or ending with "?". Allow typos (e.g., "Cn" → "Can") and informal questions (e.g., "Sermons available?"). Return only "True" or "False".
Input: "{user_input}"
        """

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=question_checker_prompt
    )

    llm_1 = ChatOpenAI(model="gpt-3.5-turbo")
    query_rewriter_chain = prompt | llm_1 | StrOutputParser()
    result = query_rewriter_chain.invoke({"user_input": user_input})
    return result


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["Appointment", "Prayer or Counselling", "Watch Sermon", "None", "Image Creation"] = Field(
        ...,
        description="Given a user question choose to route it Appointment, Prayer or Counselling, Watch Sermon, Image Creation or None.",
    )


async def question_router(user_input: str):
    system = """Classify the primary intent of the user input into exactly one of these categories: 'Image Creation', 'Appointment', 'Prayer or Counselling', 'Watch Sermon', or 'None'. Evaluate case-insensitively using these rules in order:

1. **Image Creation**: Commands to create or generate visual content, with "create", "generate", "make", "draw", or similar (e.g., "genrate", "creat") AND "image", "picture", "photo", or "art". Examples: "Generate an image of a sunset", "Create a picture of a church", "Make photo of Jesus", "Genrate art of a tree".
2. **Appointment**: Requests to manage appointments, meetings, or sessions, with "book", "schedule", "accept", "show", "delete", "read", "reserve", or similar (e.g., "shedule", "apointment") AND "appointment", "meeting", or "session". Also includes shorthand like "appt tomorrow". Examples: "Book a meeting tomorrow", "Show session details", "Delete apointment", "Can you reserve a spot for 12th May?", "appt for today".
3. **Prayer or Counselling**: Requests for spiritual or emotional support, with "pray", "prayer", "counsel", "counselling", "advice", "guidance", "talk", or similar (e.g., "preyer", "councelling"). Also includes emotional pleas like "help my soul". Examples: "Pray for my family", "I need counselling", "Give spiritual guidance", "Help me, pray now", "Talk to a pastor".
4. **Watch Sermon**: Commands to access sermons, with "watch", "show", "play", "get", "view", "stream", or typos (e.g., "paly", "shwo", "sarmon") AND "sermon", "preaching", "ministration", "message", "service", or "homily". Includes "sermon of [time]", "latest [term]", or shorthand like "sermon today". Examples: "Watch a sermon", "Play The Two Way Dance sermon", "Get yesterday’s sermon", "Show latest preaching", "Stream service now".
5. **None**: Input that is:
   - A command not matching above (e.g., "Tell a joke").
   - Too vague or single words without clear intent (e.g., "image", "appointment").
   - Questions, unless clearly implying a command (e.g., "Can you book a meeting?" → 'Appointment'). Examples: "What’s the weather?", "Tell a story", "Sermon", "Can you dance?".

**Rules**:
- Choose the first matching category in order (e.g., "Pray and book appointment" → 'Prayer or Counselling').
- Handle typos and phonetic errors (e.g., "genrate" → "generate", "appoitnment" → "appointment", "sarmon" → "sermon", "prrayer" → "prayer").
- Interpret shorthand or vague commands (e.g., "appt tomorrow" → 'Appointment', "prayer" → 'Prayer or Counselling' if implying a request).
- Allow questions implying commands to route to the relevant category (e.g., "Can you pray for me?" → 'Prayer or Counselling').
- For ambiguous inputs with "sermon" and no question words (e.g., "get sermon"), default to 'Watch Sermon'.
- Ignore filler words (e.g., "um", "please") and repetition (e.g., "pray pray").
- Output only one of: 'Image Creation', 'Appointment', 'Prayer or Counselling', 'Watch Sermon', 'None'."""
    llm = ChatOpenAI(model="gpt-4.1-mini")
    structured_llm_router = llm.with_structured_output(RouteQuery)

    if is_question_checker(user_input) == "True":
        return "None"
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_route = route_prompt | structured_llm_router
    result = await question_route.ainvoke({"question": user_input})
    return result.datasource


if __name__ == "__main__":
    async def test_inputs():
        # for i in ["Draw me a picture of Jesus", 'Who is Uche Raymond', "How do i book an appointment", "I need prayer for my husband", "Get me the sermon on Tuesday"]:
        # for i in ["Make me an image of Jesus Chrsit", "Who is Apostle Uche Raymond", "I need prayer", "Get the sermon titled the last of us"]:
        for i in ["You would like prayer for marriage."]:
            a = time.time()
            routed = await question_router(i)
            print(routed)
            b = time.time()
            print(b-a)
    asyncio.run(test_inputs())
