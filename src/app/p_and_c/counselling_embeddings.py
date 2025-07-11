from qdrant_client import AsyncQdrantClient, models
import logging
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List
from dotenv import load_dotenv
import asyncio
from app.settings import settings


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
            "Grief": ["death", "loss", "bereavement", "mourning", "sadness", "heartbroken", "grieving a loved one",
                      "funeral", "grieving process", "passed away", "why did this happen", "searching for meaning",
                      "feeling lost", "questioning God", "spiritual crisis", "grief counseling", "support group",
                      "memorial", "coping with loss", "I can't stop crying", "I feel empty"],
            "Relationships": ["marriage", "divorce", "separation", "conflict", "argument", "communication",
                              "infidelity", "affair", "breakup", "dating", "family", "children", "spouse", "partner",
                              "relationship issues", "loneliness", "intimacy", "trust", "forgiveness", "boundaries",
                              "friendship issues", "coworker conflict", "social anxiety", "romantic relationship",
                              "family dynamics", "parenting challenges", "faith-based counseling", "spiritual guidance",
                              "I'm always fighting with my partner", "I feel disconnected"],
            "Addiction": ["addiction", "substance abuse", "alcoholism", "drug abuse", "dependence", "recovery", "relapse",
                          "withdrawal", "overdose", "gambling addiction", "compulsive behavior", "craving", "support group",
                          "rehabilitation", "12-step program", "self-control", "internet addiction", "food addiction",
                          "behavioral addiction", "seeking help", "treatment options", "I can't control myself",
                          "I'm scared of what I'm doing"],
            "Anxiety": ["anxiety", "worry", "fear", "nervousness", "panic", "overthinking", "restlessness", "dread",
                        "anxious thoughts", "feeling anxious", "panic attacks", "social anxiety", "generalized anxiety",
                        "I can’t relax", "I’m constantly worried", "fear of the future", "racing thoughts",
                        "anxiety about work", "nervous breakdown"]
        }
        self.counselling_feedback = {
            "Grief": {
                "verses": ["Psalm 34:18", "Revelation 21:4", "Matthew 5:4"],
                "steps": [
                    "Allow yourself to grieve",
                    "Join a support group",
                    "Memorialize your loss"
                ],
                "counseling": "GriefShare: +2349094540644"
            },
            "Addiction": {
                "verses": ["1 Corinthians 10:13", "Philippians 4:13", "Galatians 5:1"],
                "steps": [
                    "Contact Celebrate Recovery",
                    "Remove triggers",
                    "Daily accountability"
                ],
                "counseling": "SAMHSA: +2348032235209"
            },
            "Relationships": {
                "verses": ["Colossians 3:13", "Ecclesiastes 4:9-12", "Proverbs 15:1"],
                "steps": [
                    "Practice active listening",
                    "Set healthy boundaries",
                    "Seek couples counseling"
                ],
                "counseling": "Christian Relationship Counselors Network: +2348033194594"
            },
            "Anxiety": {
                "verses": ["Philippians 4:6-7", "1 Peter 5:7", "Matthew 6:34"],
                "steps": [
                    "Pray to God",
                    "Read the word of God"
                ],
                "counseling": "Christian Anxiety Counselors Network: +2348033194594"
            },
            "Default": {
                "verses": ["Colossians 3:13", "Ecclesiastes 4:9-12", "Proverbs 15:1"],
                "steps": [
                    "Practice self-reflection",
                    "Seek spiritual guidance",
                    "Seek professional counseling"
                ],
                "counseling": "Christian Counselors Network: +2349094540644"
            }
        }
        self.similarity_threshold = 0.75
        self.collection_name = "counselling_topics"

        self.QDRANT_URL = settings.QDRANT_URL
        self.QDRANT_API_KEY = settings.QDRANT_API_KEY
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

    async def return_help(self, query: str) -> str:
        """Return counseling feedback based on the most relevant topic asynchronously."""
        if not query.strip():
            logger.warning("Empty query provided")
            return "Please provide a valid query to receive counseling feedback."
        values = await self.explain_similarity(query)
        if values["max_category_score"] < self.similarity_threshold:
            department = self.counselling_feedback["Default"]
        else:
            department = self.counselling_feedback[values['most_likely_category']]
        response_string = f"""Hello, please read these Bible verses we believe will help with your situation: {', '.join(department['verses'])}.
Please follow these steps: {', '.join(department['steps'])}.
Please call this number; they will counsel you: {department['counseling']}.
God bless you."""
        # logger.info(f"Returning help for query '{query}': {response_string}")
        return response_string

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


# if __name__ == "__main__":
#     async def test_counselling_relation():
#         try:
#             x = CounsellingRelation()
#             result = await x.return_help("He is sad he just lost his father")
#             print(result)

#         except Exception as e:
#             print(f"An error occurred: {e}")

#     asyncio.run(test_counselling_relation())
