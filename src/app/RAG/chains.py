from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os
# from langchain.schema import Document
import asyncio
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv


load_dotenv()
print(os.environ.get("TAVILY_API_KEY"))

QDRANT_URL = 'https://19df3277-f7fe-4676-95aa-8a9b7fe1568e.eu-west-2-0.aws.cloud.qdrant.io:6333'
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.K6P9M8eXXJmVl4rKMLqTc2L2EiSVs1InP78pe_J2Mws"
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
COLLECTION_NAME = "ntiem_document_collection"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


async def search_docs(query: str, k=5, fetch_k=10):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': k, 'fetch_k': fetch_k}
    )
    results = await retriever.ainvoke(query)
    # results = await vector_store.asimilarity_search(query, k=k)
    return results


# Router


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "bot_vectorstore", "web_search", "LGBTQ+_related", "GPT"] = Field(
        ...,
        description="Given a user question choose to route it to bot_vectorstore, web search, vectorstore, LGBTQ+_related, GPT.",
    )


def rag_router():
    # LLM with function call
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
#     system = """You are an AI assistant that routes questions to the right resource:
# Vectorstore: Route the question to 'vectorstore' if it is EXPLICITELY related to the following topics: New Testament International Evanglical Ministry, NTIEM, ntiem, Apostle Uche Raymond or Uche Raymond..
# Web Search: Route the question to 'web-search' if it may require current or up-to-date information for accurate result, use web-search.
# LGBTQ+_Related**: Route the question to 'LGBTQ+__related' if it pertains to LGBTQ+ topics.
# GPT: Route the question to 'GPT' if it does not fit into any of the above categories.
# Analyze the question (case-insensitive), respond with only: 'vectorstore', 'web-search', 'LGBTQ__related', or 'GPT'."""

    system = """You are an AI assistant routing questions based on these rules, evaluated in order:
- 'bot_vectorstore': Questions about the assistant’s identity, capabilities, or state, using "you" or "your" as the subject (e.g., "What’s your name?", "What can you do?"). Excludes commands or questions about other tasks (e.g., "Can you get me a sermon?"). Handles typos (e.g., "yuo", "u") and shorthand (e.g., "Who r u?"). Examples: "What is your name?", "How are you?", "What can u do?", "Who r u?", "Whats ur purpose?".
- ''vectorstore': If the question mentions "NTIEM," "ntiem," "Apostle Uche Raymond," or "Uche Raymond" (e.g., "What is NTIEM’s mission?" "Who is Uche Raymond?", "How old is New Testament International Evangelical Ministry).
- 'web-search': Route to 'web-search' if the question may require current or up-to-date information or is about people (even if not related to NTIEM, NTIEM Bot or Apostle Uche Raymond, New Testament International Evangelical Ministry").
- 'LGBTQ+__related': If the question involves LGBTQ+ topics (e.g., "What is gay marriage?").
- 'GPT': If none of the above apply (e.g., "What is gravity?").

**Instructions:**
- Check case-insensitively.
- Output only: 'bot_vectorstore', 'vectorstore', 'web-search', 'LGBTQ+__related', or 'GPT'.
- Examples: 
  - "What is your purpose?" → 'bot_vectorstore'
  - "Can you tell me about NTIEM?" → 'vectorstore'
  - "What’s the weather?" → 'web-search'
  - "Are there gay pastors?" → 'LGBTQ+__related'
  - "Tell me a story" → 'GPT'"""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    return question_router


# router_chain = rag_router()
# print(router_chain.invoke({'question': "Can you book appointments"}))
# print(router_chain.invoke(
#     {'question': "Can you get me the sermon titled test_video2"}))
# # print(router_chain.invoke(
# #     {'question': "What are the ministres of New Testament International Evangelical Minstries"}))
# print(router_chain.invoke(
#     {'question': "Whao is Apostle Uche Raymond"}))
# print(router_chain.invoke({'question': "Who is the founder of ntiem"}))
# print(router_chain.invoke({'question': "How to Dance"}))
# print(router_chain.invoke(
#     {'question': "How old is New Testament International Evangelical Ministries"}))


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def document_grader():
    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader evaluating whether a retrieved document is relevant to a user’s question.  
If the document mentions keyword(s) or has any general meaning loosely connected to the question, consider it relevant.  
This is a relaxed check—just aim to exclude clearly unrelated retrievals.  
Provide a binary ‘yes’ or ‘no’ score to show if the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    return retrieval_grader

# Generate


def answer_generator():
    # 1. Create PROPER PROMPT TEMPLATE
    prompt = ChatPromptTemplate.from_template("""
    You are NTIEM Bot a bot created by Akachi Raymond to help answer questions and provide user guidance and Christian Assistance. Use the following pieces of retrieved context to answer the question. 
    Keep the answer insightful and point detailed. Also return information you believe may be useful and provide context.

    Question: {question} 
    Context: {context} 
    Answer:
    """)

    # 2. Format documents function
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    def format_docs(docs):
        """Safely format document list into concatenated string"""
        if not isinstance(docs, list):
            raise ValueError(f"Expected list of documents, got {type(docs)}")

        return "\n\n".join(
            doc.page_content
            for doc in docs
            if hasattr(doc, 'page_content')
        )

    # 3. Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 4. Build PROPER CHAIN with correct piping
    rag_chain = (
        {"context": RunnableLambda(lambda x: format_docs(x["context"])),
         "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Hallucination Grader


def hallucination_grader():
    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    # print(hallucination_grader.invoke({"documents": docs, "generation": generation}).binary_score)
    return hallucination_grader

# Answer Grader


def answer_grader():
    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a expert grader assesser you take the context of the question and the LLM response and determine whether the LLM response addresses / resolves the question \n
            Give a binary score 'yes' or 'no'. Yes' means that the response resolves the question"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "User question: \n\n {question} \n\n LLM response: {generation}"),
        ]
    )
    # question = "Who is the current president ot the United States"
    answer_grader = answer_prompt | structured_llm_grader
    # print(answer_grader.invoke({"question": question, "generation": "The current president of the United States is Donald Trump"}).binary_score)
    return answer_grader


# Web search

# Search
# os.environ["TAVILY_API_KEY"] = "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"


# web_search_tool = TavilySearchResults(k=3)

# question = "Who is the foounder of New Testament International Evangelical Ministry"
# docs = web_search_tool.invoke({"query": question})
# web_results = "\n".join([d["content"] for d in docs])
# web_results = Document(page_content=web_results)
# print(web_results)


async def web_search_chain(query: str):
    # os.environ["TAVILY_API_KEY"] = "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"

    # Ensure API key is set
    if not os.environ.get("TAVILY_API_KEY"):
        raise ValueError("Please set the TAVILY_API_KEY environment variable.")

    # Initialize the Tavily search tool
    tavily_tool = TavilySearchResults(
        max_results=3,  # Limit to 5 results
        search_depth="advanced",  # Optional: deeper search
    )
    result = await tavily_tool.ainvoke({"query": query})
    return result

if __name__ == "__main__":
    rag_router_chain = rag_router()
    print(asyncio.run(rag_router_chain.ainvoke(
        {'question': "Can you book me an appointments?"})))
    # print(asyncio.run(web_search_chain("Grok 4")))
