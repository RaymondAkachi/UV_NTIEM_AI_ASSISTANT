from typing import List, Any, Callable, Dict, Tuple
from rapidfuzz import fuzz, process
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Replace with your preferred LLM
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from app.db_logic.models import User
from app.db_logic.database import engine
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, UTC, timedelta
from zoneinfo import ZoneInfo
import asyncio  # For testing purposes
from sqlalchemy import select
from json import loads
from app.settings import settings
from dotenv import load_dotenv
import logging
from app.p_and_c.send_number import schedule_number_send
from pymongo.errors import (
    ConnectionFailure,
    NetworkTimeout,
    OperationFailure,
    ServerSelectionTimeoutError
)
import boto3
import json
from botocore.exceptions import ClientError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_log, after_log
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Logging setup
RETRIABLE_MONGO_EXCEPTIONS = (
    ConnectionFailure,
    NetworkTimeout,
    OperationFailure,
    ServerSelectionTimeoutError
)

# MongoDB connection setup
load_dotenv()

MONGODBATLAS_URL = settings.MONGODBATLAS_URI
client = AsyncIOMotorClient(MONGODBATLAS_URL, server_api=ServerApi(
    version='1', strict=True, deprecation_errors=True))
db = client['chat_app']
chat_history = db['chat_history']

# Retry decorator for async MongoDB operations


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RETRIABLE_MONGO_EXCEPTIONS),
    before_sleep=before_log(logger, logging.INFO),
    after=after_log(logger, logging.WARNING),
    reraise=True
)
async def execute_mongo(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    """Execute an async MongoDB operation with retry logic."""
    if not asyncio.iscoroutinefunction(async_func):
        raise ValueError(
            f"Function {async_func.__name__} must be an async function")
    return await async_func(*args, **kwargs)


async def update_chat_collection(collection, user_id: str, user_msg: dict, bot_msg: dict, state: str):
    """Async function to perform the update_one operation with state."""
    return await collection.update_one(
        {"_id": user_id},
        {
            "$push": {
                "messages": {
                    "$each": [user_msg, bot_msg],
                    "$slice": -2
                }
            },
            "$set": {
                "state": state
            }
        },
        upsert=True
    )


async def update_chat_history(user_id: str, user_message: str, bot_response: str, state: str = "normal"):
    """
    Update user's chat history and conversation state, keeping only the last 2 messages.

    Args:
        user_id (str): Unique identifier for the user.
        user_message (str): The user's message content.
        bot_response (str): The bot's response content.
        state (str): The conversation state (e.g., 'normal', 'waiting_for_prayer_topic').

    Returns:
        bool: True if update is successful, False otherwise.
    """
    user_msg = {
        "sender": user_id,
        "content": user_message,
        "timestamp": datetime.now(ZoneInfo("Africa/Lagos"))
    }
    bot_msg = {
        "sender": "bot",
        "content": bot_response,
        "timestamp": datetime.now(ZoneInfo("Africa/Lagos"))
    }

    try:
        result = await execute_mongo(
            update_chat_collection,
            collection=chat_history,
            user_id=user_id,
            user_msg=user_msg,
            bot_msg=bot_msg,
            state=state
        )
        user_doc = await chat_history.find_one({"_id": user_id})
        new_count = len(user_doc["messages"]
                        ) if user_doc and "messages" in user_doc else 0
        logger.info(
            f"Chat history updated for {user_id}. Message count: {new_count}, State: {state}")
        return True
    except Exception as e:
        logger.error(f"Error updating chat history for {user_id}: {e}")
        return False


async def find_chat_history(collection, user_id: str) -> dict:
    """Async function to perform the find_one operation."""
    return await collection.find_one({"_id": user_id})


async def get_chat_history(user_id: str) -> Tuple[List[dict], str]:
    """
    Retrieve the chat history and conversation state for a given user.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        Tuple[List[dict], str]: List of messages and the current conversation state.
    """
    try:
        user_doc = await execute_mongo(
            find_chat_history,
            collection=chat_history,
            user_id=user_id
        )
        if user_doc and "messages" in user_doc:
            state = user_doc.get("state", "normal")
            logger.info(
                f"Retrieved chat history for {user_id}. Message count: {len(user_doc['messages'])}, State: {state}")
            return user_doc["messages"], state
        logger.info(
            f"No chat history found for {user_id}, returning default state 'normal'")
        return [], "normal"
    except Exception as e:
        logger.error(f"Error retrieving chat history for {user_id}: {e}")
        return [], "normal"


async def get_prayer_obj(user_input: str):
    prayer_prompt = ("""You are an AI assistant tasked with determining the user's intent based on their input regarding prayer requests. The church offers the following prayer options:
1. Marriage  
2. Career  
3. Finances  
4. Health  
5. Children  
6. Direction  
7. Spiritual Attack  
8. Others

The user has been prompted with:'Please select out of these options what exactly you would like prayer for.\n\n1. Marriage\n2. Career\n3. Finances\n4. Health\n5. Children\n6. Direction\n7. Spiritual Attack\n8. Others'  
Based on the user's response, identify which options they are selecting. The user might input numbers (e.g., "1", "3"), category names (e.g., "Marriage", "Health"), or a combination (e.g., "1. Marriage", "Career and 3"). Your task is to extract the selected options and return their corresponding numbers in the order they appear in the input. If the input does not indicate a selection from these options or seems unrelated, return null.
Instructions

Case-Insensitive Matching: Convert the input to lowercase.  
Number Detection: Use regex (e.g., r'\b[1-8]\b') to find standalone numbers 1–8, recording their positions and values.  
Category Detection: Search for exact category names (e.g., r'\bmarriage\b', r'\bspiritual attack\b') in the input, recording their positions and corresponding numbers.  
Order Preservation: Sort matches by their positions in the input.  
Unique Selections: Collect category numbers in the order they first appear, avoiding duplicates (e.g., "Marriage and Marriage" returns [1], not [1,1]).  
Output: If selections are found, return a list of numbers; otherwise, return null.

Response Format
Always return a JSON object with the key "selected_options" containing the list of numbers or null.  

Example: {{"selected_options": [1, 3]}}  
Example: {{"selected_options": null}}

Examples

Input: "Marriage" → {{"selected_options": [1]}}
Input: "1. Marriage" → {{"selected_options": [1]}}  
Input: "3 and 1" → {{"selected_options": [3, 1]}}  
Input: "Career and Health" → {{"selected_options": [2, 4]}}  
Input: "Hello world" → {{"selected_options": null}}
""")

    # Create the PromptTemplate
    prayer_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prayer_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4.1-mini")
    chain = prayer_q_prompt | llm | StrOutputParser()

    result = await chain.ainvoke({'input': str(user_input)})
    result = json.loads(result)
    return_list = []

    prayer_list = await get_json_object('prayer_list')
    prayer_info = await get_json_object('prayer_info')
    if not result['selected_options']:
        return None

    for i in result['selected_options']:
        category = prayer_list[str(i)]
        return_list.append(prayer_info[category])

    return return_list


async def get_json_object(key):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_BUCKET_SECRET_ACCESS_KEY
        )
        response = s3_client.get_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key="prayer_feedback.json"
        )
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data.get(key, {})

    except ClientError as e:
        logger.error(
            "Something went wroong when trying to retrieve data from prayer_feedback.json in ntiembotbucket")
        return {}
    except Exception as e:
        logger.error(
            "Something went wroong when trying to retrieve data from prayer_feedback.json in ntiembotbucket")
        return {}


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


async def rewriters_func(chat_history: list, user_question_1: str, state: str):
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
    results['state'] = state
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
    chat_history, state = await get_chat_history(user_phone_number)
    if state != "normal":
        total_res = ''
        prayer_objects = await get_prayer_obj(user_question)
        if not prayer_objects:
            if chat_history == []:
                await add_user_to_db(user_name, user_phone_number)
                result = await rewriters_func(chat_history, user_question, state)
            else:
                result = await rewriters_func(chat_history, user_question, state)
            result['state'] = 'normal'
            return result
        else:
            for i in prayer_objects:
                total_res += total_res + f"{i['prayer']}\n\n"
                await schedule_number_send(user_phone_number, i['contact'])

            return {'answerable': True, "Response": total_res, "state": "normal"}

    if chat_history == []:
        await add_user_to_db(user_name, user_phone_number)
        result = await rewriters_func(chat_history, user_question, state)
    else:
        result = await rewriters_func(chat_history, user_question, state)
    result['state'] = 'normal'
    return result


# if __name__ == "__main__":
    # x = asyncio.run(update_chat_history("2349094540644", "Hello I need prayer",
    #                 "Please what exactly do you need prayer for?", "prayer"))

    # async def update_chat_history(user_id: str, user_message: str, bot_response: str, state: str = "normal"):
    # async def test():
    #     history = ['User: I want prayer',
    #                "Bot: Please select out of these options what exactly you would like prayer for.\n\n\n1. Marriage\n\n2. Health\n\n3. Children\n\n4. Direction\n\n5. Others"]
    #     result = await chain.ainvoke({'chat_history': history, "input": "User: Health"})
    #     print(result)
    # asyncio.run(test())
if __name__ == "__main__":
    async def update_test():
        res_1 = await query_rewrite('Who is Apostle Uche Raymond', "Akachi", "2349094540644")
        print(res_1)
    asyncio.run(update_test())
