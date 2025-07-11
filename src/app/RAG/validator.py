from qdrant_client import AsyncQdrantClient, models
import logging
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List
# import time # Use for in-scipt testing
from app.settings import settings
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicValidator:
    """Validates if a query aligns with allowed topics using Qdrant for async similarity search."""

    def __init__(self):
        """Initialize the validator with Qdrant and OpenAI embeddings."""
        # Validate environment variables
        # if not os.getenv("OPENAI_API_KEY"):
        #     logger.error("OPENAI_API_KEY environment variable is not set")
        #     raise ValueError("OPENAI_API_KEY is required")
        # if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        #     logger.error("QDRANT_URL and QDRANT_API_KEY environment variables are required")
        #     raise ValueError("QDRANT_URL and QDRANT_API_KEY are required")

        self.embeddings_model = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY)
        self.allowed_topics: Dict[str, List[str]] = {
            "stress_management": ["stress", "anxiety", "burnout", "work pressure", "sad", "depressed", "anxious", "stress",
                                  "pain", "confused", "afraid", "sick", "hurt", "scared", "angry", "enraged",
                                  "deliverance", "breakthrough", "healing"],
            "spiritual_growth": ["prayer", "faith", "scripture study", "meditation", "Jesus", "Bible", "christianity",
                                 "scripture", "Gospel", "rapture"],
            "relationships": ["marriage", "family conflict", "parenting", "friendship"],
            "people_of_God": ["Apostle", "Pastor", "Preacher", "Minister", "church", "evangelism", "ministry", "Gospel"],
            "Jews": ["Abraham", "Moses", "Judaism", "Covenant", "Jews"],
            "News": ["Nigeria", "News", "Politics", "President", "Policy", "Judiciary", "Bill", "Senator"],
            "Greetings": ["Hello", "Hi", "Good day"]
        }
        self.blocklist = [
            "lgbtq+", "gay", "lesbian", "transgender", "queer", "bisexual"
        ]
        self.similarity_threshold = 0.79
        self.collection_name = "topics"
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
        if loop.is_running():
            loop.create_task(self._initialize_collection())
        else:
            loop.run_until_complete(self._initialize_collection())

    async def _initialize_collection(self):
        """Initialize Qdrant collection asynchronously."""
        if not await self._collection_exists():
            logger.info(
                f"Collection '{self.collection_name}' does not exist. Creating and populating...")
            await self._create_and_populate_collection()
        else:
            logger.info(
                f"Collection '{self.collection_name}' exists. Verifying compatibility...")
            await self._verify_collection()

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

            # Prepare texts for batch embedding (synchronous)
            all_texts = [category for category in self.allowed_topics] + \
                        [kw for keywords in self.allowed_topics.values()
                         for kw in keywords]
            all_payloads = [(cat, cat) for cat in self.allowed_topics] + \
                           [(cat, kw) for cat, kws in self.allowed_topics.items()
                            for kw in kws]

            # Batch embed texts (synchronous)
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

            # Upsert points (async)
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

            # Per-category maximum similarity
            category_scores = {}
            for category in self.allowed_topics.keys():
                result = await self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="category", match=models.MatchValue(value=category))]
                    ),
                    limit=1
                )
                category_scores[category] = result[0].score if result else 0.0

            # Average of category max scores
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

    def contains_blocked_terms(self, query: str) -> bool:
        """Check if query contains blocked terms."""
        query_lower = query.lower()
        return any(term in query_lower for term in self.blocklist)

    async def is_topic_allowed(self, query: str) -> bool:
        """Determine if query matches allowed topics semantically asynchronously."""
        if not query.strip():
            logger.warning("Empty query provided")
            return False
        if self.contains_blocked_terms(query):
            logger.info(f"Query '{query}' contains blocked terms")
            return False
        scores = await self.calculate_similarity(query)
        allowed = scores["max_score"] >= self.similarity_threshold
        logger.info(
            f"Query '{query}' is {'allowed' if allowed else 'rejected'} with max_score {scores['max_score']}")
        return allowed

    async def explain_similarity(self, query: str) -> dict:
        """Provide detailed similarity analysis asynchronously."""
        if not query.strip():
            logger.warning("Empty query provided")
            return {"query": query, "threshold": self.similarity_threshold, "decision": "rejected", "scores": {}}
        scores = await self.calculate_similarity(query)
        explanation = {
            "query": query,
            "threshold": self.similarity_threshold,
            "decision": "allowed" if scores["max_score"] >= self.similarity_threshold else "rejected",
            "scores": scores
        }
        logger.info(f"Similarity explanation for '{query}': {explanation}")
        return explanation

    async def add_topic_category(self, category: str, keywords: List[str]):
        """Add new topic category and keywords to Qdrant at runtime asynchronously."""
        if not category or not keywords:
            logger.error("Category and keywords cannot be empty")
            raise ValueError("Category and keywords are required")
        try:
            self.allowed_topics[category] = keywords
            # Batch embed category and keywords (synchronous)
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


if __name__ == "__main__":
    try:
        async def test_validator():
            a = time.time()
            validator = TopicValidator()
            res1 = await validator.explain_similarity("Hello how are you")
            res2 = await validator.explain_similarity("Who is winning the war between isreal and iran?")
            b = time.time()
            print(b-a)
            print(res1)
            print(res2)
        asyncio.run(test_validator())
    except BaseException as e:
        print(f"An Error Occured: {e}")
