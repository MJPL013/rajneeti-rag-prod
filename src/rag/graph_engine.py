"""
GraphRAG Engine — Intent-Driven Graph Traversal.

Architecture:
  1. Vector grounding  : Embed query -> Neo4j vector index -> candidate statement_ids
  2. Graph traversal   : Given intent + grounded ids, run specialized Cypher
  3. Context assembly   : Return structured RAGContext to the Generator

Traversal Strategies:
  - get_temporal_evolution  : Timeline of stance changes
  - get_media_portrayal     : Left vs Center vs Right coverage
  - get_politician_persona  : High-weight rhetoric & promises
  - get_factual_context     : Default flat retrieval (reranked)

Singleton pattern preserved for Streamlit compatibility.
"""

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Optional

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import GraphDBError
from src.models.schema import (
    RAGIntent,
    RAGContext,
    TimelineEntry,
    MediaGroup,
)


class GraphRAGEngine:
    """
    GraphRAG Engine — singleton.
    Provides vector grounding + intent-driven graph traversal.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphRAGEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing GraphRAG Engine (Intent-Driven)...")

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
            logger.info(f"Connected to Neo4j -- {settings.NEO4J_URI}")
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

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def is_connected(self) -> bool:
        return self.driver is not None

    def retrieve(
        self,
        query: str,
        filters: Dict[str, Any] = None,
    ) -> List[Dict]:
        """
        Legacy flat retrieval — vector search + rerank.
        Kept for backwards-compatibility with the old orchestrator.
        """
        return self._vector_search_rerank(query, filters)

    def expand_context(self, top_results: List[Dict]) -> str:
        """Legacy context expansion — kept for backwards-compatibility."""
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
                with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                    result = session.run(cypher, aid=aid)
                    record = result.single()
                    if record:
                        header = f"\n--- Article: {record['title']} ---\n"
                        header += f"Source: {record['source']} | Date: {record['publish_date']}\n"
                        header += f"URL: {record['url']}\n"
                        body = "\n".join(record["statements"])
                        full_context.append(header + body)
            except Exception as e:
                logger.error(f"Failed to expand context for article {aid}: {e}")
                continue

        return "\n\n".join(full_context)

    # ==================================================================
    # VECTOR GROUNDING (Step 1 of the new pipeline)
    # ==================================================================

    def vector_ground(self, query: str, top_k: int = 20) -> List[str]:
        """
        Embed the query, hit the Neo4j vector index, return grounded
        statement_ids (no reranking — that is done post-traversal).
        """
        if not self.is_connected():
            raise GraphDBError("Neo4j driver is not connected.")

        query_vector = self.embedding_model.encode(query).tolist()

        cypher = """
        CALL db.index.vector.queryNodes('statement_embeddings', $k, $embedding)
        YIELD node, score
        RETURN node.statement_id AS statement_id, score
        ORDER BY score DESC
        """

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, embedding=query_vector, k=top_k)
                return [record["statement_id"] for record in result if record["statement_id"]]
        except Exception as e:
            logger.error(f"Vector grounding failed: {e}")
            raise GraphDBError(f"Vector grounding failed: {e}") from e

    # ==================================================================
    # INTENT-DRIVEN TRAVERSAL STRATEGIES (Step 2)
    # ==================================================================

    def get_temporal_evolution(
        self,
        politician: str,
        statement_ids: List[str],
    ) -> RAGContext:
        """
        Timeline traversal:
          Match grounded statements -> TimeFrame
          Group by year_month, ordered chronologically.
        """
        cypher = """
        MATCH (p:Politician {name: $politician})<-[:FOCUSES_ON]-(a:Article)
              -[:CONTAINS_STATEMENT]->(s:Statement)
        WHERE s.statement_id IN $sids

        OPTIONAL MATCH (a)-[:PUBLISHED_IN]->(tf:TimeFrame)
        OPTIONAL MATCH (a)-[:PUBLISHED_BY]->(src:Source)

        WITH tf.year_month AS ym,
             s.text AS stmt,
             src.name AS source_name
        ORDER BY ym ASC

        RETURN ym,
               collect(DISTINCT stmt) AS statements,
               collect(DISTINCT source_name) AS sources
        ORDER BY ym ASC
        """

        timeline = []
        all_sources = set()

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, politician=politician, sids=statement_ids)
                for record in result:
                    ym = record["ym"] or "Unknown"
                    stmts = record["statements"] or []
                    srcs = record["sources"] or []
                    timeline.append(TimelineEntry(
                        year_month=ym,
                        statements=stmts,
                        sources=srcs,
                    ))
                    all_sources.update(srcs)
        except Exception as e:
            logger.error(f"Temporal traversal failed: {e}")
            raise GraphDBError(f"Temporal traversal failed: {e}") from e

        total = sum(len(t.statements) for t in timeline)

        return RAGContext(
            intent=RAGIntent.TEMPORAL_EVOLUTION,
            politician=politician,
            timeline=timeline,
            total_statements=total,
            sources_used=list(all_sources),
        )

    def get_media_portrayal(
        self,
        politician: str,
        statement_ids: List[str],
    ) -> RAGContext:
        """
        Media contrast traversal:
          Group statements by Source.political_leaning.
          Filter for perspective = 'About Politician' when available.
        """
        cypher = """
        MATCH (p:Politician {name: $politician})<-[:FOCUSES_ON]-(a:Article)
              -[:PUBLISHED_BY]->(src:Source)
        MATCH (a)-[:CONTAINS_STATEMENT]->(s:Statement)
        WHERE s.statement_id IN $sids

        WITH src.political_leaning AS leaning,
             src.name AS source_name,
             s.text AS stmt

        RETURN leaning,
               collect(DISTINCT source_name) AS source_names,
               collect(DISTINCT stmt) AS statements
        ORDER BY leaning
        """

        media_groups = []
        all_sources = set()

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, politician=politician, sids=statement_ids)
                for record in result:
                    leaning = record["leaning"] or "unknown"
                    src_names = record["source_names"] or []
                    stmts = record["statements"] or []
                    media_groups.append(MediaGroup(
                        leaning=leaning,
                        source_names=src_names,
                        statements=stmts,
                    ))
                    all_sources.update(src_names)
        except Exception as e:
            logger.error(f"Media portrayal traversal failed: {e}")
            raise GraphDBError(f"Media portrayal traversal failed: {e}") from e

        total = sum(len(g.statements) for g in media_groups)

        return RAGContext(
            intent=RAGIntent.MEDIA_CONTRAST,
            politician=politician,
            media_groups=media_groups,
            total_statements=total,
            sources_used=list(all_sources),
        )

    def get_politician_persona(
        self,
        politician: str,
        statement_ids: List[str],
    ) -> RAGContext:
        """
        Persona traversal:
          High-weight rhetoric — the politician's own voice.
          Filter: perspective = 'By Politician' OR classification = 'Rhetoric'
                  OR weight > 2.0
        """
        cypher = """
        MATCH (s:Statement)-[:MADE_BY]->(p:Politician {name: $politician})
        WHERE s.statement_id IN $sids

        OPTIONAL MATCH (a:Article)-[:CONTAINS_STATEMENT]->(s)
        OPTIONAL MATCH (a)-[:PUBLISHED_BY]->(src:Source)

        RETURN s.text AS stmt,
               s.weight AS weight,
               s.classification AS classification,
               src.name AS source_name
        ORDER BY s.weight DESC
        """

        rhetoric = []
        high_weight = []
        all_sources = set()

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, politician=politician, sids=statement_ids)
                for record in result:
                    stmt = record["stmt"]
                    w = record["weight"] or 0
                    cls = record["classification"] or ""
                    src = record["source_name"]

                    if src:
                        all_sources.add(src)

                    if cls == "Rhetoric":
                        rhetoric.append(stmt)
                    if w and w > 2.0:
                        high_weight.append(stmt)
        except Exception as e:
            logger.error(f"Persona traversal failed: {e}")
            raise GraphDBError(f"Persona traversal failed: {e}") from e

        # If persona-specific traversal yielded nothing, fall back to
        # a broader search using the grounded IDs directly
        if not rhetoric and not high_weight:
            rhetoric, high_weight, all_sources = self._persona_fallback(
                politician, statement_ids
            )

        total = len(set(rhetoric + high_weight))

        return RAGContext(
            intent=RAGIntent.PERSONA,
            politician=politician,
            rhetoric_statements=rhetoric,
            high_weight_statements=high_weight,
            total_statements=total,
            sources_used=list(all_sources),
        )

    def get_factual_context(
        self,
        query: str,
        politician: Optional[str] = None,
        filters: Dict[str, Any] = None,
    ) -> RAGContext:
        """
        Default flat retrieval — vector search + rerank.
        Returns a RAGContext with flat_statements populated.
        """
        results = self._vector_search_rerank(query, filters)

        stmts = []
        all_sources = set()

        for res in results:
            stmts.append({
                "text": res["document"],
                "source": res["metadata"].get("source", ""),
                "politician": res["metadata"].get("politician", ""),
                "publish_date": str(res["metadata"].get("publish_date", "")),
                "title": res["metadata"].get("title", ""),
                "score": res["score"],
            })
            src = res["metadata"].get("source")
            if src:
                all_sources.add(src)

        return RAGContext(
            intent=RAGIntent.FACTUAL,
            politician=politician or "",
            flat_statements=stmts,
            total_statements=len(stmts),
            sources_used=list(all_sources),
        )

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    def _persona_fallback(
        self, politician: str, statement_ids: List[str]
    ):
        """
        Broader persona fallback when MADE_BY edges are sparse.
        Uses FOCUSES_ON instead.
        """
        cypher = """
        MATCH (p:Politician {name: $politician})<-[:FOCUSES_ON]-(a:Article)
              -[:CONTAINS_STATEMENT]->(s:Statement)
        WHERE s.statement_id IN $sids

        OPTIONAL MATCH (a)-[:PUBLISHED_BY]->(src:Source)

        RETURN s.text AS stmt,
               s.weight AS weight,
               s.classification AS classification,
               src.name AS source_name
        ORDER BY s.weight DESC
        LIMIT 30
        """

        rhetoric = []
        high_weight = []
        all_sources = set()

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, politician=politician, sids=statement_ids)
                for record in result:
                    stmt = record["stmt"]
                    w = record["weight"] or 0
                    cls = record["classification"] or ""
                    src = record["source_name"]

                    if src:
                        all_sources.add(src)

                    if cls == "Rhetoric":
                        rhetoric.append(stmt)
                    elif w and w > 2.0:
                        high_weight.append(stmt)
                    else:
                        # Include anyway if traversal is thin
                        rhetoric.append(stmt)
        except Exception as e:
            logger.error(f"Persona fallback failed: {e}")

        return rhetoric, high_weight, all_sources

    def _vector_search_rerank(
        self,
        query: str,
        filters: Dict[str, Any] = None,
    ) -> List[Dict]:
        """
        Vector search on Neo4j statement_embeddings index + CrossEncoder rerank.
        """
        if not self.is_connected():
            raise GraphDBError("Neo4j driver is not connected.")

        query_vector = self.embedding_model.encode(query).tolist()

        sentiment_filter = filters.get("sentiment") if filters else None
        classification_filter = filters.get("classification") if filters else None

        cypher = """
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
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(
                    cypher,
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

        # Rerank
        pairs = [[query, res["document"]] for res in raw_results]
        scores = self.reranker.predict(pairs)

        scored_results = []
        for i, res in enumerate(raw_results):
            scored_results.append({
                "id": res.get("article_id", ""),
                "document": res["document"],
                "metadata": {
                    "article_id": res.get("article_id", ""),
                    "politician": res.get("politician", ""),
                    "source": res.get("source", ""),
                    "publish_date": str(res.get("publish_date", "")),
                    "url": res.get("url", ""),
                    "title": res.get("title", ""),
                    "sentiment": res.get("sentiment", ""),
                    "classification": res.get("classification", ""),
                    "summary": res.get("summary", ""),
                },
                "score": float(scores[i]),
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[: settings.TOP_K_RERANK]

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")
