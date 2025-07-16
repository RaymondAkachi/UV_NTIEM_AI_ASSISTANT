from qdrant_client import AsyncQdrantClient, models
import logging
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from .send_number import schedule_number_send
from typing import Dict, List
from dotenv import load_dotenv
import asyncio
from app.settings import settings
import boto3
import json
from botocore.exceptions import ClientError


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounsellingRelation:
    """Validates counseling-related queries and provides feedback using Qdrant for async similarity search."""

    def __init__(self):
        """Initialize the validator with Qdrant and OpenAI embeddings."""
        # Validate environment variables
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY is required")
        # if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        #     logger.error("QDRANT_URL and QDRANT_API_KEY environment variables are required")
        #     raise ValueError("QDRANT_URL and QDRANT_API_KEY are required")

        self.embeddings_model = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY)
        self.allowed_topics: Dict[str, List[str]] = {
            "Marriage": ["marriage", "spouse", "husband", "wife", "partner", "marital harmony", "love and unity",
                         "marital issues", "communication in marriage", "intimacy", "trust in marriage", "infidelity",
                         "reconciliation", "strengthening our bond", "family unity", "marriage restoration",
                         "spousal support", "divorce prevention", "healthy relationship"],
            "Career": ["job", "work", "employment", "career path", "professional growth", "job opportunity",
                       "workplace success", "career advancement", "promotion", "job stability", "work challenges",
                       "career guidance", "professional skills", "job satisfaction", "work-life balance", "new job",
                       "career transition", "success at work"],
            "Finances": ["financial breakthrough", "debt relief", "financial stability", "money problems", "provision",
                         "poverty", "unemployment", "financial struggles", "prosperity", "economic hardship",
                         "financial blessing", "budget management", "income increase", "financial security",
                         "business success", "financial wisdom", "debt cancellation"],
            "Health": ["healing", "illness", "sickness", "disease", "mental health", "physical health", "recovery",
                       "medical condition", "surgery", "health challenge", "chronic illness", "pain relief",
                       "emotional healing", "wellness", "strength and health", "medical treatment",
                       "restoration of health", "healing miracle"],
            "Children": ["children", "kids", "family", "parenting", "child's future", "protection for children",
                         "children's health", "guidance for kids", "child's education", "children's safety",
                         "raising children", "child behavior", "teen challenges", "blessing for children",
                         "child's faith", "family harmony", "children's success"],
            "Direction": ["guidance", "God's will", "direction in life", "clarity", "purpose", "decision making",
                          "path forward", "divine guidance", "life choices", "wisdom", "spiritual direction",
                          "finding purpose", "life path", "God's plan", "discernment", "clear path", "life decisions"],
            "Spiritual_Attack": ["village people"]
        }
        self.similarity_threshold = 0.76
        self.collection_name = "counselling_topics"

        self.QDRANT_URL = settings.QDRANT_URL
        self.QDRANT_API_KEY = settings.QDRANT_API_KEY
        self.default_entries = {
            "i need counselling",
            "want counselling",
            "counselling",
            "need counselling",
            "counselling for me",
            "please counselling",
            "request counselling",
            "counselling request",
            "i want counselling",
            "in need of counselling",
            "ask for counselling",
            "seeking counselling"
        }

        # Initialize Async Qdrant client
        try:
            self.client = AsyncQdrantClient(
                url=self.QDRANT_URL,
                api_key=self.QDRANT_API_KEY,
                timeout=30
            )
            logger.info("Connected to Qdrant Cloud (async)")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

        # Determine vector size (synchronous)
        try:
            dummy_embedding = self.embeddings_model.embed_query("dummy")
            self.vector_size = len(dummy_embedding)
        except Exception as e:
            logger.error(f"Failed to compute dummy embedding: {str(e)}")
            raise

        # Initialize collection (async)
        loop = asyncio.get_event_loop()
        try:
            if loop.is_running():
                loop.create_task(self._initialize_collection())
            else:
                loop.run_until_complete(self._initialize_collection())
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    async def _initialize_collection(self):
        """Initialize Qdrant collection asynchronously."""
        try:
            if not await self._collection_exists():
                logger.info(
                    f"Collection '{self.collection_name}' does not exist. Creating and populating...")
                await self._create_and_populate_collection()
            else:
                logger.info(
                    f"Collection '{self.collection_name}' exists. Verifying compatibility...")
                await self._verify_collection()
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    async def _collection_exists(self) -> bool:
        """Check if the Qdrant collection exists asynchronously."""
        try:
            collections = await self.client.get_collections()
            return any(collection.name == self.collection_name for collection in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            raise

    async def _verify_collection(self):
        """Verify that the existing collection has the correct vector size asynchronously."""
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            if collection_info.config.params.vectors.size != self.vector_size:
                logger.error(f"Collection '{self.collection_name}' has vector size {collection_info.config.params.vectors.size}, "
                             f"but expected {self.vector_size}")
                raise ValueError(
                    "Incompatible vector size in existing collection")
            logger.info(
                f"Collection '{self.collection_name}' is compatible with vector size {self.vector_size}")
        except Exception as e:
            logger.error(f"Error verifying collection: {str(e)}")
            raise

    async def _create_and_populate_collection(self):
        """Create Qdrant collection and populate with topic embeddings asynchronously."""
        try:
            # Create collection
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE)
            )
            logger.info(f"Created collection '{self.collection_name}'")

            # Prepare texts for batch embedding
            all_texts = [category for category in self.allowed_topics] + \
                        [kw for keywords in self.allowed_topics.values()
                         for kw in keywords]
            all_payloads = [(cat, cat) for cat in self.allowed_topics] + \
                           [(cat, kw) for cat, kws in self.allowed_topics.items()
                            for kw in kws]

            # Batch embed texts
            embeddings = self.embeddings_model.embed_documents(all_texts)

            # Create points
            points = [
                models.PointStruct(
                    id=i,
                    vector=emb,
                    payload={"category": cat, "text": text}
                )
                for i, (emb, (cat, text)) in enumerate(zip(embeddings, all_payloads))
            ]

            # Upsert points
            await self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                f"Populated collection '{self.collection_name}' with {len(points)} points")
        except Exception as e:
            logger.error(f"Failed to create and populate collection: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> list:
        """Convert user query to embedding vector (synchronous)."""
        try:
            return self.embeddings_model.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query '{query}': {str(e)}")
            raise

    async def calculate_similarity(self, query: str) -> dict:
        """Calculate similarity scores against allowed topics in Qdrant asynchronously."""
        try:
            query_embedding = self._get_query_embedding(query)

            # Overall maximum similarity
            overall_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=1
            )
            max_score = overall_result[0].score if overall_result else 0.0

            # Per-category average similarity
            category_scores = {}
            for category in self.allowed_topics.keys():
                result = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="category", match=models.MatchValue(value=category))]
                    ),
                    limit=10
                )
                scores = [r.score for r in result] if result else [0.0]
                category_scores[category] = sum(
                    scores) / len(scores) if scores else 0.0

            # Average of category scores
            average_score = sum(category_scores.values()) / \
                len(category_scores) if category_scores else 0.0

            return {
                "max_score": max_score,
                "average_score": average_score,
                "category_scores": category_scores
            }
        except Exception as e:
            logger.error(
                f"Error calculating similarity for query '{query}': {str(e)}")
            return {"max_score": 0.0, "average_score": 0.0, "category_scores": {}}

    async def is_topic_allowed(self, query: str) -> bool:
        """Determine if query matches allowed topics semantically asynchronously."""
        if not query.strip():
            logger.warning("Empty query provided")
            return False
        scores = await self.calculate_similarity(query)
        max_category_score = max(
            scores["category_scores"].values()) if scores["category_scores"] else 0.0
        allowed = max_category_score >= self.similarity_threshold
        logger.info(
            f"Query '{query}' is {'allowed' if allowed else 'rejected'} with max_category_score {max_category_score}")
        return allowed

    async def explain_similarity(self, query: str) -> dict:
        """Provide detailed similarity analysis asynchronously."""
        if not query.strip():
            logger.warning("Empty query provided")
            return {
                "query": query,
                "threshold": self.similarity_threshold,
                "decision": "not_found",
                "most_likely_category": "Default",
                "max_category_score": 0.0,
                "scores": {}
            }
        scores = await self.calculate_similarity(query)
        max_category_score = max(
            scores["category_scores"].values()) if scores["category_scores"] else 0.0
        decision = "found" if max_category_score >= self.similarity_threshold else "not_found"
        most_likely_category = max(
            scores["category_scores"], key=scores["category_scores"].get) if scores["category_scores"] else "Default"
        explanation = {
            "query": query,
            "threshold": self.similarity_threshold,
            "decision": decision,
            "most_likely_category": most_likely_category,
            "max_category_score": max_category_score,
            "scores": scores
        }
        logger.info(f"Similarity explanation for '{query}': {scores}")
        return explanation

    async def return_help(self, query: str, user_number: str) -> str:
        """Return counseling feedback based on the most relevant topic asynchronously."""
        if not query.strip():
            logger.warning("Empty query provided")
            return "Please provide a valid query to receive counseling feedback."
        values = await self.explain_similarity(query)
        if values["max_category_score"] < self.similarity_threshold or query in self.default_entries:
            prayer, number = await self._load_counselling_info("Others")
        else:
            prayer, number = await self._load_counselling_info(values['most_likely_category'])
        number = "Please reach out to our counsellor at: " + number
        await schedule_number_send(user_number, number)
        return prayer

    async def _load_counselling_info(self, category) -> Dict:
        """Load prayer feedback from S3."""
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
            counsellor_number = data.get('counsellor_number', '2349094540644')
            counselling_info = data.get('counselling_info', {})
            return counselling_info.get(category, {}), counsellor_number

        except ClientError as e:
            logger.error(f"Failed to load prayer feedback from S3: {str(e)}")
            return "verses: Proverbs 22:6, Psalm 127:3, Deuteronomy 6:6-7\n\nPrayer: Father Lord, help me to pray to you in spirit and in truth, in Jesus' name, Amen.", "2349094540644"
        except Exception as e:
            logger.error(
                f"Unexpected error loading prayer feedback from S3: {str(e)}")
            return "verses: Proverbs 22:6, Psalm 127:3, Deuteronomy 6:6-7\n\nPrayer: Father Lord, help me to pray to you in spirit and in truth, in Jesus' name, Amen.", "2349094540644"

    async def add_topic_category(self, category: str, keywords: List[str]):
        """Add new topic category and keywords to Qdrant at runtime asynchronously."""
        if not category or not keywords:
            logger.error("Category and keywords cannot be empty")
            raise ValueError("Category and keywords are required")
        try:
            self.allowed_topics[category] = keywords
            # Batch embed category and keywords
            all_texts = [category] + keywords
            embeddings = self.embeddings_model.embed_documents(all_texts)
            current_count = (await self.client.count(collection_name=self.collection_name)).count
            points = [
                models.PointStruct(
                    id=current_count + i,
                    vector=emb,
                    payload={"category": category, "text": text}
                )
                for i, (emb, text) in enumerate(zip(embeddings, all_texts))
            ]
            await self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                f"Added category '{category}' with {len(keywords)} keywords to Qdrant")
        except Exception as e:
            logger.error(f"Failed to add category '{category}': {str(e)}")
            raise


class NewCounsellingRelation:
    def __init__(self):
        # self.request = request
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.prompt = PromptTemplate(
            input_variables=["user_message"],
            template="""
Generate a Christian prayer and supporting Bible verses based on a user request: {user_message}. 
The response should align strictly with traditional Christian beliefs and biblical truths. It should explicitly avoid supporting any modern practices or ideologies that contradict these teachings, such as LGBTQ+ lifestyles.
""")
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def get_prayer(self, request):
        result = await self.chain.ainvoke({'user_message': request})
        additional_text = "Kindly meditate on these: while we provide you with a number to speak with our counsellor.\n\n"
        result = additional_text + str(result)

        return result

    async def return_help(self, query, phone_number):
        result = await self.get_prayer(query)
        user_number = phone_number
        counsellor_number = await self._get_counsellor_number()
        text = f"Here is our counsellor's number: {counsellor_number}"
        additional_data = {"priority": "high"}

        success = await schedule_number_send(user_number, text, additional_data)
        if not success:
            print("Failed to schedule task.")

        return result

    async def _get_counsellor_number(self):
        """Load prayer feedback from S3."""
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
            return data.get('counsellor_number', '')

        except ClientError as e:
            logger.error(f"Failed to counsellor number from S3: {str(e)}")
            return "Something went wrong: Reach a counsellor here: 2349094540644"

        except Exception as e:
            logger.error(
                f"Failed to counsellor number from S3: {str(e)}")
            return "Something went wrong: Reach a counsellor here: 2349094540644"

        # Add functionality functionality to send message after 2 number after 2 minutes


if __name__ == "__main__":
    async def test_counselling_relation():
        try:
            x = CounsellingRelation()
            result = await x.return_help("I need counselling on my marriage", "2349094540644")
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")

    asyncio.run(test_counselling_relation())
