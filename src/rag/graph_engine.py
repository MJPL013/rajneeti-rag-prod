"""
GraphRAG Engine — Neo4j-based retrieval with vector search and reranking.

Responsibilities:
  - Connect to Neo4j (env-driven)
  - Perform vector search via Cypher
  - Rerank results
  - Expand context via graph traversal

Does NOT contain LLM logic.
"""
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import GraphDBError


class GraphRAGEngine:
    """
    GraphRAG Engine — singleton.
    Queries Neo4j for statement retrieval with vector search + reranking.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphRAGEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing GraphRAG Engine...")

        # --- Validate required config ---
        if not settings.NEO4J_URI:
            raise GraphDBError("NEO4J_URI is not configured. Set it in .env.")
        if not settings.NEO4J_PASSWORD:
            raise GraphDBError("NEO4J_PASSWORD is not configured. Set it in .env.")

        # --- Neo4j Connection ---
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            )
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j — {settings.NEO4J_URI}")
        except Exception as e:
            raise GraphDBError(f"Neo4j connection failed ({settings.NEO4J_URI}): {e}") from e

        # --- Embedding Model ---
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        try:
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                trust_remote_code=True,
            )
        except Exception as e:
            raise GraphDBError(f"Failed to load embedding model: {e}") from e

        # --- Reranker ---
        logger.info(f"Loading reranker model: {settings.RERANKING_MODEL_NAME}")
        try:
            self.reranker = CrossEncoder(settings.RERANKING_MODEL_NAME)
        except Exception as e:
            raise GraphDBError(f"Failed to load reranker model: {e}") from e

        logger.info("GraphRAG Engine initialized.")

    def is_connected(self) -> bool:
        """Check if Neo4j connection is active."""
        return self.driver is not None

    def retrieve(
        self,
        query: str,
        filters: Dict[str, Any] = None,
    ) -> List[Dict]:
        """
        Retrieves relevant statements from Neo4j using vector search,
        then reranks with CrossEncoder.
        """
        if not self.is_connected():
            raise GraphDBError("Neo4j driver is not connected.")

        # Embed the query
        query_vector = self.embedding_model.encode(query).tolist()

        # Build filter params
        sentiment_filter = filters.get("sentiment") if filters else None
        classification_filter = filters.get("classification") if filters else None

        cypher_query = """
        CALL db.index.vector.queryNodes('statement_embeddings', $k, $embedding)
        YIELD node, score
        
        WHERE ($sentiment IS NULL OR node.sentiment = $sentiment)
          AND ($classification IS NULL OR node.classification = $classification)
        
        MATCH (article:Article)-[:CONTAINS_STATEMENT]->(node)
        OPTIONAL MATCH (article)-[:FOCUSES_ON]->(politician:Politician)
        OPTIONAL MATCH (article)-[:PUBLISHED_BY]->(source:Source)
        
        RETURN 
            node.text AS document,
            node.summary AS summary,
            score,
            node.sentiment AS sentiment,
            node.classification AS classification,
            article.title AS title,
            article.article_id AS article_id,
            article.url AS url,
            article.publish_date AS publish_date,
            politician.name AS politician,
            source.name AS source
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher_query,
                    embedding=query_vector,
                    k=settings.TOP_K_RETRIEVAL,
                    sentiment=sentiment_filter,
                    classification=classification_filter,
                )
                raw_results = [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            raise GraphDBError(f"Neo4j query failed: {e}") from e

        if not raw_results:
            logger.warning("No documents found in GraphRAG.")
            return []

        # Rerank using CrossEncoder
        pairs = [[query, res["document"]] for res in raw_results]
        scores = self.reranker.predict(pairs)

        scored_results = []
        for i, res in enumerate(raw_results):
            scored_results.append(
                {
                    "id": res.get("article_id", ""),
                    "document": res["document"],
                    "metadata": {
                        "article_id": res.get("article_id", ""),
                        "politician": res.get("politician", ""),
                        "source": res.get("source", ""),
                        "publish_date": res.get("publish_date", ""),
                        "url": res.get("url", ""),
                        "title": res.get("title", ""),
                        "sentiment": res.get("sentiment", ""),
                        "classification": res.get("classification", ""),
                        "summary": res.get("summary", ""),
                    },
                    "score": float(scores[i]),
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[: settings.TOP_K_RERANK]

    def expand_context(self, top_results: List[Dict]) -> str:
        """Fetches full article context for the top results using graph traversal."""
        if not top_results or not self.is_connected():
            return ""

        unique_article_ids = []
        seen = set()
        for res in top_results:
            aid = res["metadata"].get("article_id")
            if aid and aid not in seen:
                unique_article_ids.append(aid)
                seen.add(aid)
            if len(unique_article_ids) >= 2:
                break

        logger.info(f"GraphRAG: expanding context for articles: {unique_article_ids}")

        full_context = []

        for aid in unique_article_ids:
            cypher = """
            MATCH (a:Article {article_id: $aid})-[:CONTAINS_STATEMENT]->(s:Statement)
            OPTIONAL MATCH (a)-[:FOCUSES_ON]->(p:Politician)
            OPTIONAL MATCH (a)-[:PUBLISHED_BY]->(src:Source)
            RETURN 
                a.title AS title,
                a.url AS url,
                a.publish_date AS publish_date,
                src.name AS source,
                collect(s.text) AS statements
            """
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, aid=aid)
                    record = result.single()

                    if record:
                        article_header = f"\n--- Article: {record['title']} ---\n"
                        article_header += f"Source: {record['source']} | Date: {record['publish_date']}\n"
                        article_header += f"URL: {record['url']}\n"
                        article_body = "\n".join(record["statements"])
                        full_context.append(article_header + article_body)
            except Exception as e:
                logger.error(f"Failed to expand context for article {aid}: {e}")
                continue

        return "\n\n".join(full_context)

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")
