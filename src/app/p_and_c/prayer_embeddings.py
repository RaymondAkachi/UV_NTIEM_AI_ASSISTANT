from qdrant_client import AsyncQdrantClient, models
import logging
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List
import asyncio
from app.settings import settings
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrayerRelation:
    """Validates prayer-related queries and provides feedback using Qdrant for async similarity search."""

    def __init__(self):
        """Initialize the validator with Qdrant and OpenAI embeddings."""
        # Validate environment variables
        # if not os.getenv("OPENAI_API_KEY"):
        #     logger.error("OPENAI_API_KEY environment variable is not set")
        #     raise ValueError("OPENAI_API_KEY is required")
        # if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        #     logger.error(
        #         "QDRANT_URL and QDRANT_API_KEY environment variables are required")
        #     raise ValueError("QDRANT_URL and QDRANT_API_KEY are required")

        self.embeddings_model = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY)
        self.allowed_topics: Dict[str, List[str]] = {
            "Marriage": ["marriage", "spouse", "partner", "husband", "wife", "marital relationship",
                         "infertility", "intimacy", "pregnancy", "parenting within marriage",
                         "communication with spouse", "marital issues", "spousal support"],
            "Health": ["Healing", "illness", "disease", "sickness", "Mental Health", "injury", "surgery",
                       "physical health", "medical condition", "health recovery", "health challenge"],
            "Anxiety": ["Fear", "Worry", "Apprehension", "anxiety-induced stress", "Uncertainty",
                        "panic attacks", "generalized anxiety", "social anxiety"],
            "Deliverance": ["Principalities and Powers", "strange spirits", "ancestral spirits", "Village people"],
            "Finances": ["Debt", "Financial struggles", "Poverty", "Unemployment", "Financial insecurity"],
            "Favour": ["God's favour", "Job opportunities", "Business success", "professional networking",
                       "business connections", "Promotions", "Career advancements", "industry recognition"]
        }
        self.prayer_feedback = {
            "Marriage": {
                "verses": ["1 Corinthians 13:4-7", "Ephesians 4:2-3", "Proverbs 3:3-4"],
                "prayer": "Lord, strengthen this union with patience and understanding. Help them love as You love. Amen.",
                "contact": "Church Marriage Helpline: 1-800-327-4357"
            },
            "Health": {
                "verses": ["Jeremiah 30:17", "3 John 1:2", "Psalm 103:2-3"],
                "prayer": "Heavenly Father, bring restoration and wholeness. Guide medical staff and comfort the afflicted. Amen.",
                "contact": "Church Health Prayer Helpline: 1-888-230-2637"
            },
            "Anxiety": {
                "verses": ["Philippians 4:6-7", "1 Peter 5:7", "Matthew 6:34"],
                "prayer": "Prince of Peace, calm troubled hearts. Help them cast all cares on You. Amen.",
                "contact": "Church Anxiety Helpline: 1-800-950-6264"
            },
            "Deliverance": {
                "verses": ["Psalm 107:2", "2 Corinthians 1:10", "Psalm 34:17"],
                "prayer": "Father God, deliver me from any existing powers working against me, in Jesus' name, Amen.",
                "contact": "Deliverance Ministers. Call: 1-800-A-FAMILY"
            },
            "Finances": {
                "verses": ["Deuteronomy 8:18", "Malachi 3:10", "Philippians 4:19"],
                "prayer": "Dear Heavenly Father, I thank you for your promise to bless me financially. Help me trust in your provision and seek your Kingdom first, in Jesus' name, Amen.",
                "contact": "Finance intercessors helpline: 1-800-A-FAMILY"
            },
            "Favour": {
                "verses": ["Psalm 5:12", "Luke 2:52", "Proverbs 3:3-4"],
                "prayer": "Heavenly Father, I ask for your favour to rest upon me, in Jesus' name, Amen.",
                "contact": "Favour intercessors: 1-800-A-FAMILY"
            },
            "Default": {
                "verses": ["Proverbs 22:6", "Psalm 127:3", "Deuteronomy 6:6-7"],
                "prayer": "Father Lord, help me to pray to you in spirit and in truth, in Jesus' name, Amen.",
                "contact": "Focus on the prayer: 1-800-A-FAMILY"
            }
        }
        self.similarity_threshold = 0.75
        self.collection_name = "prayer_topics"
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
            return {"query": query, "threshold": self.similarity_threshold, "decision": "not_found",
                    "most_likely_category": "Default", "max_category_score": 0.0, "scores": {}}
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
        """Return prayer feedback based on the most relevant topic asynchronously."""
        values = await self.explain_similarity(query)
        if values["decision"] != "found":
            department = self.prayer_feedback["Default"]
        else:
            department = self.prayer_feedback[values['most_likely_category']]
        response_string = f"""Hello, please read these Bible verses we believe will help with your situation: {', '.join(department['verses'])}.
Also, pray this prayer: {department['prayer']}
Please call this number; they will pray with you: {department['contact']}
While waiting for their response, please read the Bible verses provided and pray. God bless you."""
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


if __name__ == "__main__":
    async def test_prayer_relation():
        try:
            x = PrayerRelation()
            result = await x.return_help("I need prayer for my depression")
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")

    asyncio.run(test_prayer_relation())


# No LONGER BEING USED
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os

# load_dotenv('.env')

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# class PrayerRelation:
#     def __init__(self):
#         self.embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#         self.allowed_topics = {
#             "Marriage": ["marriage", "spouse", "partner", "husband", "wife", "marital relationship",
#                          "infertility", "intimacy", "pregnancy", "parenting within marriage",
#                          "communication with spouse", "marital issues", "spousal support"],
#             "Health": ["Healing", "illness", "disease", "sickness", "Mental Health", "injury", "surgery",
#                        "physical health", "medical condition", "health recovery", "health challenge"],
#             "Anxiety": ["Fear", "Worry", "Apprehension", "anxiety-induced stress", "Uncertainty",
#                         "panic attacks", "generalized anxiety", "social anxiety"],
#             "Deliverance": ["Principalities and Powers", "strange spirits", "ancestral spirits", "Village people"],
#             "Finances": ["Debt", "Financial struggles", "Poverty", "Unemployment", "Financial insecurity"],
#             "Favour": ["God's favour", "Job opportunities", "Business success", "professional networking",
#                        "business connections", "Promotions", "Career advancements", "industry recognition"]
#         }

#         self.prayer_feedback = {
#             "Marriage": {
#                 "verses": ["1 Corinthians 13:4-7", "Ephesians 4:2-3", "Proverbs 3:3-4"],
#                 "prayer": "Lord, strengthen this union with patience and understanding. Help them love as You love. Amen.",
#                 "contact": "Church Marriage Helpline: 1-800-327-4357"
#             },
#             "Health": {
#                 "verses": ["Jeremiah 30:17", "3 John 1:2", "Psalm 103:2-3"],
#                 "prayer": "Heavenly Father, bring restoration and wholeness. Guide medical staff and comfort the afflicted. Amen.",
#                 "contact": "Church Health Prayer Helpline: 1-888-230-2637"
#             },
#             "Anxiety": {
#                 "verses": ["Philippians 4:6-7", "1 Peter 5:7", "Matthew 6:34"],
#                 "prayer": "Prince of Peace, calm troubled hearts. Help them cast all cares on You. Amen.",
#                 "contact": "Church Anxiety Helpline: 1-800-950-6264"
#             },
#             "Deliverance": {
#                 "verses": ["Psalm 107:2", "2 Corithians 1:10", "Psalm 34:17"],
#                 "prayer": "Father God deliver me from any existing powers working against mem, in Jesus' name, Amen",
#                 "contact": "Deliverance Ministers. Call: 1-800-A-FAMILY"
#             },
#             "Finances": {
#                 "verses": ["Deuteronomy 8:18", "Malachi 3:10", "Phillipians 4:19"],
#                 "prayer": "Dar Heavenly Father, I thn=ank you for your promise to bless me financially. Help me trust in your provision and seek your Kingdom first, in Jesus' name Amen",
#                 "contact": "Finance intercessors helpline: 1-800-A-FAMILY"
#             },
#             "Favour": {
#                 "verses": ["Psalm 5:12", "Luke 2:52", "Proverbs 3:3-4"],
#                 "prayer": "Heavely Father I ask for your Favour to rest upon me",
#                 "contact": "Favour intercessors: 1-800-A-FAMILY"
#             },
#             "Default": {
#                 "verses": ["Proverbs 22:6", "Psalm 127:3", "Deuteronomy 6:6-7"],
#                 "prayer": "Father Lord help me to pray to you in spirit and in truth in Jesus' name , Amen",
#                 "contact": "Focus on the prayer: 1-800-A-FAMILY"
#             }
#         }
#         self.topic_embeddings, self.category_indices = self._precompute_embeddings()
#         self.similarity_threshold = 0.75  # Adjusted for average similarity

#     def _precompute_embeddings(self):
#         """Convert allowed topics to embeddings and store category indices"""
#         embeddings = []
#         category_indices = {}
#         idx = 0
#         for category, keywords in self.allowed_topics.items():
#             embeddings.append(self.embeddings_model.embed_query(
#                 category))  # Category name
#             start_idx = idx
#             idx += 1
#             for keyword in keywords:
#                 embeddings.append(
#                     self.embeddings_model.embed_query(keyword))  # Keywords
#                 idx += 1
#             end_idx = idx
#             category_indices[category] = (start_idx, end_idx)
#         return np.array(embeddings), category_indices

#     def _get_query_embedding(self, query: str) -> np.ndarray:
#         """Convert user query to embedding vector"""
#         return np.array(self.embeddings_model.embed_query(query))

#     def calculate_similarity(self, query: str) -> dict:
#         """Calculate similarity scores against all allowed topics"""
#         query_embedding = self._get_query_embedding(query)
#         similarities = cosine_similarity(
#             [query_embedding], self.topic_embeddings)[0]

#         category_scores = {}
#         for category, (start, end) in self.category_indices.items():
#             category_similarities = similarities[start:end]
#             # Average similarity per category
#             category_scores[category] = np.mean(category_similarities)

#         return {
#             "max_score": np.max(similarities).item(),
#             "average_score": np.mean(similarities).item(),
#             "category_scores": category_scores
#         }

#     def is_topic_allowed(self, query: str) -> bool:
#         """Determine if query matches allowed topics semantically"""
#         scores = self.calculate_similarity(query)
#         max_category_score = max(scores["category_scores"].values())
#         return max_category_score >= self.similarity_threshold

#     def explain_similarity(self, query: str) -> dict:
#         """Provide detailed similarity analysis"""
#         scores = self.calculate_similarity(query)
#         max_category_score = max(scores["category_scores"].values())
#         decision = "found" if max_category_score >= self.similarity_threshold else "not_found"
#         most_likely_category = max(
#             scores["category_scores"], key=scores["category_scores"].get)
#         explanation = {
#             "query": query,
#             "threshold": self.similarity_threshold,
#             "decision": decision,
#             "most_likely_category": most_likely_category,
#             "max_category_score": max_category_score,
#             "scores": scores
#         }
#         return explanation

#     def return_help(self, query: str) -> str:
#         values = self.explain_similarity(query)
#         if values["decision"] == 'found':
#             department = self.prayer_feedback[str(
#                 values['most_likely_category'])]
#         else:
#             department = self.prayer_feedback["Default"]
#         response_string = f"""Hello Please read these bible verses we believe it will help with your situation {str(department['verses'])}
# And also prayer this prayer: {str(department['prayer'])}
# Please call this number they will prayer with you: {str(department['contact'])}
# While waiting for thier response please read the bible verses provided and pray, God bless you."""
#         return response_string

#     def add_topic_category(self, category: str, keywords: list):
#         """Add new topic category at runtime"""
#         self.allowed_topics[category] = keywords
#         new_embeddings = [self.embeddings_model.embed_query(category)]
#         new_embeddings.extend(
#             [self.embeddings_model.embed_query(kw) for kw in keywords])
#         self.topic_embeddings = np.vstack(
#             [self.topic_embeddings, new_embeddings])
#         start_idx = len(self.topic_embeddings) - len(new_embeddings)
#         end_idx = len(self.topic_embeddings)
#         self.category_indices[category] = (start_idx, end_idx)


# Example usage
# validator = PrayerRelation()
# queries = [
#     "I'm struggling with my marriage",
#     "Hi, Explain quantum field theory"
# ]

# for query in queries:
#     result = validator.explain_similarity(query)
#     print(f"Query: {query}")
#     print(f"Decision: {result['decision']}")
#     print(f"Most Likely Category: {result['most_likely_category']}")
#     print(f"Max Category Score: {result['max_category_score']:.2f}")
#     print("Category Scores:")
#     for cat, score in result['scores']['category_scores'].items():
#         print(f"- {cat}: {score:.2f}")
#     print("\n")
