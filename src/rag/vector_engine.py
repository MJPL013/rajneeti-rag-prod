"""
VectorRAG Engine — ChromaDB-based retrieval with embedding search and reranking.

Responsibilities:
  - Connect to ChromaDB persistent store
  - Perform vector similarity search
  - Rerank results with CrossEncoder
  - Expand context by fetching full articles
"""
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import VectorDBError


class VectorRAGEngine:
    """
    VectorRAG Engine — singleton.
    Queries ChromaDB for statement retrieval with embedding search + reranking.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorRAGEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing VectorRAG Engine...")

        # --- ChromaDB Client ---
        db_path = str(settings.VECTOR_DB_DIR)
        try:
            self.client = chromadb.PersistentClient(path=db_path)
        except Exception as e:
            raise VectorDBError(f"Failed to open ChromaDB at '{db_path}': {e}") from e

        # --- Embedding Function ---
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        try:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
        except Exception as e:
            raise VectorDBError(f"Failed to load embedding model: {e}") from e

        # --- Collection ---
        try:
            self.collection = self.client.get_collection(
                name="politician_statements",
                embedding_function=self.embedding_fn,
            )
            logger.info(
                f"VectorDB collection 'politician_statements' loaded "
                f"({self.collection.count()} documents)."
            )
        except Exception as e:
            raise VectorDBError(
                f"Collection 'politician_statements' not found. Run ingestion first. Error: {e}"
            ) from e

        # --- Reranker ---
        logger.info(f"Loading reranker model: {settings.RERANKING_MODEL_NAME}")
        try:
            self.reranker = CrossEncoder(settings.RERANKING_MODEL_NAME)
        except Exception as e:
            raise VectorDBError(f"Failed to load reranker model: {e}") from e

        logger.info("VectorRAG Engine initialized.")

    def retrieve(self, query: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Retrieves relevant statements from ChromaDB, reranks them.
        """
        logger.info(f"VectorRAG retrieving for: {query} | filters: {filters}")

        # Build where clause
        where_clause = {}
        if filters:
            conditions = []
            for k, v in filters.items():
                if v:
                    conditions.append({k: {"$eq": v}})

            if len(conditions) > 1:
                where_clause = {"$and": conditions}
            elif len(conditions) == 1:
                where_clause = conditions[0]

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=settings.TOP_K_RETRIEVAL,
                where=where_clause if where_clause else None,
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise VectorDBError(f"ChromaDB query failed: {e}") from e

        if not results["documents"][0]:
            logger.warning("No documents found in VectorRAG.")
            return []

        # Rerank
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        retrieved_ids = results["ids"][0]

        pairs = [[query, doc] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)

        scored_results = []
        for i in range(len(scores)):
            scored_results.append(
                {
                    "id": retrieved_ids[i],
                    "document": retrieved_docs[i],
                    "metadata": retrieved_metas[i],
                    "score": float(scores[i]),
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[: settings.TOP_K_RERANK]

    def expand_context(self, top_results: List[Dict]) -> str:
        """
        Fetches full article context for the top results.
        Takes the top 2 unique articles and fetches all their statements.
        """
        if not top_results:
            return ""

        unique_article_ids = []
        seen = set()
        for res in top_results:
            aid = res["metadata"]["article_id"]
            if aid not in seen:
                unique_article_ids.append(aid)
                seen.add(aid)
            if len(unique_article_ids) >= 2:
                break

        logger.info(f"VectorRAG: expanding context for articles: {unique_article_ids}")

        full_context = []

        for aid in unique_article_ids:
            try:
                article_stmts = self.collection.get(where={"article_id": aid})
            except Exception as e:
                logger.error(f"Failed to expand context for article {aid}: {e}")
                continue

            stmts = article_stmts["documents"]
            metas = article_stmts["metadatas"]

            if not stmts:
                continue

            first_meta = metas[0]
            article_header = f"\n--- Article: {first_meta.get('title', 'Unknown')} ---\n"
            article_header += f"Source: {first_meta.get('source', 'Unknown')} | Date: {first_meta.get('publish_date', 'Unknown')}\n"
            article_header += f"URL: {first_meta.get('url', 'Unknown')}\n"

            article_body = "\n".join(stmts)
            full_context.append(article_header + article_body)

        return "\n\n".join(full_context)
