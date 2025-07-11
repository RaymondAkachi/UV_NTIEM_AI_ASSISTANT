from .chains import search_docs, document_grader, answer_generator, web_search_chain
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.runnables import RunnableConfig
from .state import GraphState
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from app.settings import settings


load_dotenv()

QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "ntiem_bot_docs"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ntiem_bot_docs",
    embedding=embeddings,
)


async def retrieve_about_bot(state: GraphState):
    question = state['question']
    docs = await vector_store.asimilarity_search(query=question, k=5)
    documents = []
    for doc in docs:
        documents.append(doc)
    return {'documents': documents, 'question': question, "inner_source": "bot_vectorstore"}


async def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = await search_docs(query=question)
    return {"documents": documents, "question": question, "inner_source": "vectorstore"}


async def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    inner_source = state['inner_source']

    # RAG generation
    chain = answer_generator()
    generation = await chain.ainvoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}


async def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    chain = document_grader()
    filtered_docs = []
    for d in documents:
        score = await chain.ainvoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}


async def web_search(state: GraphState):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    # docs = await web_search_chain(query=question)
    # web_results = "\n".join([d["content"] for d in docs])
    # web_results = Document(page_content=web_results)

    # return {"documents": [web_results], "question": question, "inner_source": "web_search"}
    documents = []
    docs = await web_search_chain(query=question)
    for doc in docs:
        web_result = Document(page_content=doc['content'])
        documents.append(web_result)

    return {"documents": documents, "question": question, "inner_source": "web_search"}


def LGBTQ_handler(state: GraphState):
    question = state['question']
    generation = f"This question which is ({question}) not be answered because it is lgbtq+ related.\nPlease ask a question about NTIEM ministry or christianity"

    return {"inner_source": "LGTBQ", "generation": generation, "question": question}


def answer_rag(state: GraphState):
    generation = state['generation']
    question = state['question']
    inner_source = state['inner_source']

    TRIGGER_WORDS = ["sad", "depressed", "anxious", "stress",
                     "pain", "addict", "afraid", "sick", "hurt",
                     "scared", "angry", "enraged", 'deliverance', "breakthrough",
                     "healing", "grief", "marriage"]
    needs_help = any(word in question.lower() for word in TRIGGER_WORDS)
    if inner_source == 'web_search':
        generation = "This information is not from us but sourced from the web the answers you get might not be entirely accurate.\n" + generation

    if needs_help:
        generation = generation + \
            "\nWould you like prayer or counselling? Please choose one. If you are not interested ignore the question"

    return {"generation": generation, "question": question, "inner_source": inner_source}


def reject_question(state: GraphState):
    question = state['question']

    generation = f"""
    Your question: ({question}) is outside my use case.\n
    Please ask a question more specific to NTIEM ministry and christianity"""

    return {'generation': generation}


def no_documents(state: GraphState):
    question = state['question']
    generation = f"Sorry we could not find any information on your question({question}) please try another question"

    return {"generation": generation}


async def llm_fallback(state: GraphState):
    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Prompt
    system = """You are the helper bot of New Testament International Evangelical Ministries named NTIEM Bot.
    you are a christian whose role is to help users address and find information related to Jesus Christ and his church.
    You are to anlways answer a question from a CHRISTIAN perspective. Your creator is Akachi Raymond and your apostlic Leader is Apostle Uche Raymond."""
    llm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Using your knowlegde provide an answer to my question: \n\n {question} \n Keep your response concise.",
            ),
        ]
    )

    question = state['question']

    llm_answer = llm_prompt | llm | StrOutputParser()
    generation = await llm_answer.ainvoke({"question": question})

    TRIGGER_WORDS = ["sad", "depressed", "anxious", "stress",
                     "pain", "confused", "afraid", "sick", "hurt",
                     "scared", "angry", "enraged", 'deliverance', "breakthrough",
                     "healing"]

    needs_help = any(word in question.lower() for word in TRIGGER_WORDS)
    additional_info = "This information is not from us.\n"
    generation = additional_info + generation

    if needs_help:
        generation = generation + \
            "\nWould you like prayer or counselling? Please choose one. If you are not interested ignore the question"

    return {"generation": generation}


def no_answer(state: GraphState):
    question = state['question']
    generation = state['generation']
    additional_information = generation
    generation = "Information may not be entirely precise but hope it helps.\n"+generation

    return {"generation": generation}


def web_decision(state: GraphState):
    return {"inner_source": "web_search"}


def gpt_decision(state: GraphState):
    return {"inner_source": "GPT"}
