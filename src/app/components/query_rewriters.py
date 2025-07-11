from typing import List
from rapidfuzz import fuzz, process
# from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Replace with your preferred LLM
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from models import User
from database import engine
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, UTC
# import asyncio  # For testing purposes
from sqlalchemy import select
from json import loads
from datetime import datetime, timedelta
from settings import settings
from dotenv import load_dotenv

load_dotenv('.env')

####### GET QUESTION HISTORY #######
# MongoDB connection setup
MONGODBATLAS_URL = settings.MONGODBATLAS_URI
client = AsyncIOMotorClient(MONGODBATLAS_URL, server_api=ServerApi(
    version='1', strict=True, deprecation_errors=True))

# Select database and collection
db = client['chat_app']
chat_history = db['chat_history']


async def update_chat_history(user_id, user_message, bot_response):
    """
    Update user's chat history, keeping only the last 20 messages.
    Uses $slice to automatically remove oldest messages as needed.

    :param user_id: Unique identifier for the user
    :param user_message: The user's message content
    :param bot_response: The bot's response content
    """
    # New messages to add
    user_msg = {
        "sender": user_id,
        "content": user_message,
        "timestamp": datetime.now(UTC)
    }
    bot_msg = {
        "sender": "bot",
        "content": bot_response,
        "timestamp": datetime.now(UTC)
    }

    # Append new messages and cap at 2 using $slice
    result = await chat_history.update_one(
        {"_id": user_id},
        {
            "$push": {
                "messages": {
                    "$each": [user_msg, bot_msg],  # Add both messages
                    "$slice": -2                # Keep only the last 2 messages
                }
            }
        },
        upsert=True
    )

    # Estimate new count (for logging purposes)
    user_doc = await chat_history.find_one({"_id": user_id})
    # new_count = len(user_doc["messages"]
    #                 ) if user_doc and "messages" in user_doc else 0
    print(f"Chat history updated for {user_id}")


async def get_chat_history(user_id):
    """
    Retrieve the chat history for a given user.

    :param user_id: Unique identifier for the user
    :return: List of messages or an empty list if user not found or no messages exist
    """
    user_doc = await chat_history.find_one({"_id": user_id})
    if user_doc and "messages" in user_doc:
        return user_doc["messages"]
    return []


contextualize_q_system_prompt = (
    """Given a chat history and the latest user input, determine if the input can be answered using only the chat history (answer explicitly or implicitly present). Output a JSON object with:
- "answerable": Boolean (true if answer in chat history, false otherwise).
- "Response": String (direct answer if answerable; reformulated standalone input if unanswerable, preserving form as question or statement/command, ensuring questions are answerable without chat history).

**Rules**:
- **Answerable**: Input is a question or a command requesting chat history information (e.g., "Tell me my last question"), and the answer is in the chat history.
- **Not Answerable**: Input is an action command (e.g., booking, canceling), a question lacking chat history information, or a capability question without explicit evidence.
- Answerable: Provide concise answer from chat history.
- Not Answerable: 
  - Questions: Reformulate as a standalone question answerable without chat history (e.g., avoid specific references like "yesterday").
  - Statements/Commands: Reformulate as a standalone statement/command.
  - No explanatory text.
- Handle typos (e.g., "apointment" → "appointment", "sarmon" → "sermon"), shorthand (e.g., "appt" → "appointment"), vague inputs, fillers (e.g., "um"), and repetition. Case-insensitive.
- Questions start with "what", "when", "who", "can", "is", or end with "?". Others are statements/commands.
- Output only JSON object.

**Examples**:
1. Chat History: [Human: I booked an appointment for 12th May 2024. AI: Confirmed at 2pm.]
   Input: What’s my appt time?
   Output: {{"answerable": true, "Response": "2pm"}}

2. Chat History: [Human: Who is Elon musk. AI: He is the owner of X and other companies.]
   Input: Who are his kids?
   Output: {{"answerable": true, "Response": "Who are Elon Musk's kids?"}}

3. Chat History: [Human: I need the sermon of yesterday. AI: Here's the link: <link>.]
   Input: Who preached it?
   Output: {{"answerable": false, "Response": "Who preached the sermon of yesterday?"}} 

4. Chat History: []
   Input: Book an apointment for tommorow
   Output: {{"answerable": false, "Response": "I want to book an appointment for tomorrow"}} 

5. Chat History: []
   Input: Can you book apointments?
   Output: {{"answerable": false, "Response": "Can you book appointments?"}} 

6. Chat History: [Human: What’s your name? AI: I’m Grok.]
   Input: Tell me my last question
   Output: {{"answerable": true, "Response": "What’s your name?"}} 

Output only the JSON object."""
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

llm_1 = ChatOpenAI(model="gpt-4.1-mini")
chain = contextualize_q_prompt | llm_1 | StrOutputParser()


# Define the prompt template
prompt_template = """
**Instructions:**  
Given a user's question, rewrite it according to the following rules:  
1. **Questions About NTIEM Bot, ntiem bot**:
If the question is about "NTIEM Bot" or "ntiem bot"(e.g., "What is NTIEM Bot's name?" "How does ntiem bot work"), rewrite it to use "you" or "your" to refer to the assistant as grammatically appropriate.

2. **Church-Related Questions:**  
   - If the question is about a church, ministry, institution, gathering(e.g., asking about service times, location, events, mission) and does *not* mention a specific church name, rewrite it to refer to "New Testament International Evangelical Ministry."  
   - If the question already mentions a specific church name (e.g., "Dance Battle Church," "First Baptist Church"), leave it unchanged.  

2. **Leadership-Related Questions:**  
   - If the question refers to a leadership role (e.g., "leader," "Apostle," "founder," "supervisor," "pastor") and does *not* include a specific person's name, rewrite it to refer to "Apostle Uche Raymond" as the subject, incorporating his role and the church name "New Testament International Evangelical Ministry" for context.  
   - If the question already includes a specific person's name (e.g., "Pastor John," "Reverend Smith"), leave it unchanged.  

**Additional Guidelines:**  
-Apply the rules in the order presented: first check for questions about NTIEM Bot, then church-related questions, then leadership-related questions.
-Each applicable rule can modify the question sequentially.
-Maintain the form of a question in the rewritten output.
-Do not answer the question; only rewrite it according to the rules.
-Provide only the rewritten question as output, without additional text.

**Examples:**
- **Input:** "What is NTIEM Bot's name?"  
  **Output:** "What is your name?"  
- **Input:** "When is ntiem bot answering?"  
  **Output:** "When are you answering?"
-**Input:** "Who are you, what is your name"
  **Output:** "Who are you what is your name"
- **Input:** "What are the service times?"  
  **Output:** "What are the service times at New Testament International Evangelical Ministry?"  
- **Input:** "What are the service times at Battle Church?"  
  **Output:** "What are the service times at Battle Church?"  
- **Input:** "Who is the leader?"  
  **Output:** "Who is Apostle Uche Raymond"  
- **Input:** "What is the founder's vision?"  
  **Output:** "What is Apostle Uche Raymond's vision as the founder of New Testament International Evangelical Ministry?"  
- **Input:** "Tell me about Pastor John."  
  **Output:** "Tell me about Pastor John."
- **Input:** "Who is the founder of John Wick"  
  **Output:** "Who is the founder of John Wick."
- **Input:** "Who is the founder"  
  **Output:** "Who is Apostle Uche Raymond" 
**User Question:**  
{user_question}

**Rewritten Question:**
"""


# Create the PromptTemplate
prompt = PromptTemplate(
    input_variables=["user_question"],
    template=prompt_template
)


llm_2 = ChatOpenAI(model="gpt-4.1-mini")


query_rewriter_chain = prompt | llm_2 | StrOutputParser()
chain_comb = chain | query_rewriter_chain


def convert_relative_date(relative_day: str) -> str:
    """
    Convert 'today', 'tomorrow', or 'yesterday' to the format '12th May 2024' using current date.

    Args:
        relative_day (str): One of 'today', 'tomorrow', or 'yesterday'.

    Returns:
        str: Date in format 'day{suffix} month year' (e.g., '12th May 2024').

    Raises:
        ValueError: If relative_day is not 'today', 'tomorrow', or 'yesterday'.
    """
    current_date = datetime.now()  # Use system’s current date
    day_map = {'today': 0, 'tomorrow': 1, 'yesterday': -1}

    if relative_day.lower() not in day_map:
        raise ValueError("Input must be 'today', 'tomorrow', or 'yesterday'")

    target_date = current_date + timedelta(days=day_map[relative_day.lower()])
    day = target_date.day
    month = target_date.strftime('%B')
    year = target_date.year

    suffix = 'th' if 10 <= day % 100 <= 20 else {
        1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {month} {year}"


def replace_relative_dates(query: str, target_words: List[str] = None, threshold: float = 85) -> str:
    """
    Process a query, detect 'yesterday', 'tomorrow', or 'today' (with typos), and replace with formatted dates.

    Args:
        query (str): Input query (e.g., "I have an apointment tommorow").
        target_words (List[str], optional): Words to match. Defaults to ["yesterday", "tomorrow", "today"].
        threshold (float): Fuzzy match similarity threshold (0-100, default 80).

    Returns:
        str: Query with matched words replaced by formatted dates (e.g., "14th May 2025").
    """
    if target_words is None:
        target_words = ["yesterday", "tomorrow", "today"]

    words = query.split()
    result = []

    for word in words:
        match = process.extractOne(
            word, target_words, scorer=fuzz.ratio, score_cutoff=threshold)
        if match:
            matched_word, _, _ = match
            try:
                formatted_date = convert_relative_date(matched_word)
                result.append(formatted_date)
            except ValueError:
                result.append(word)  # Keep original if conversion fails
        else:
            result.append(word)

    return " ".join(result)


async def rewriters_func(chat_history: list, user_question_1: str):
    history = []
    user_question = replace_relative_dates(user_question_1)
    if chat_history:
        for msg in chat_history:
            sender = msg['sender']
            content = msg['content']
            if msg['sender'] != 'bot':
                sender = f"User:{sender}"
            history.append(f"{sender}: {content}")
    res = await chain.ainvoke({"chat_history": history, "input": user_question})
    results = loads(res)
    if not results['answerable']:
        user_question = results['Response']
        response = await query_rewriter_chain.ainvoke(
            {'user_question': user_question})
        results['Response'] = response
    return results


async def add_user_to_db(user_name, phone_number):
    async with AsyncSession(engine) as session:
        # Check if phone number already exists
        query = select(User).where(User.phone_number == str(phone_number))
        result = await session.execute(query)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            print(f"User with phone number {phone_number} already exists")
        else:
            new_user = User(name=str(user_name),
                            phone_number=str(phone_number))
            session.add(new_user)
            await session.commit()
            print(
                f"New user named {user_name} with phone number {phone_number} created")


async def query_rewrite(user_question, user_name, user_phone_number):
    chat_history = await get_chat_history(user_phone_number)
    if chat_history == []:
        await add_user_to_db(user_name, user_phone_number)
        result = await rewriters_func(chat_history, user_question)
    else:
        result = await rewriters_func(chat_history, user_question)
    return result


# if __name__ == "__main__":
#     async def update_test():
#         res_1 = await query_rewrite('What is happening today?', "Akachi", "2349094540644")
#         print(res_1)
#     asyncio.run(update_test())
