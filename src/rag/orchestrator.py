"""
RAG Orchestrator — Intent-Driven Router (Dynamic Config).

Pipeline:
  1. Intent Classification  : LLM classifies query -> (intent, entities, filters)
  2. Vector Grounding       : Embed query -> top-N statement_ids from Neo4j
  3. Graph Traversal        : Route to strategy based on intent
  4. Generation             : Format RAGContext -> intent-specific prompt -> LLM answer
  5. Response Assembly      : Return structured RAGResponse

Dynamic Config:
  - Accepts llm_provider / api_key / model_name overrides at construction.
  - Streamlit creates a new Orchestrator when the user changes provider.
"""

import json
from typing import Dict, Any, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import ConfigurationError, LLMGenerationError
from src.models.schema import (
    RAGIntent,
    QueryAnalysis,
    RAGContext,
    RAGResponse,
)


# ==================================================================
# INTENT CLASSIFIER PROMPT
# ==================================================================

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a political news analysis system about Indian politics.
Analyze the user's question and extract structured signals.

AVAILABLE INTENTS:
- TEMPORAL_EVOLUTION: When the user asks about changes over time, stance evolution, historical shifts.
  Examples: "How has Modi's stance on farm laws changed?", "Trace Mamata's position on NRC from 2019 to 2023"
- MEDIA_CONTRAST: When the user asks about media bias, coverage differences, media portrayal.
  Examples: "How do left vs right media cover Yogi?", "Compare NDTV and ThePrint coverage of Kejriwal"
- PERSONA: When the user asks about a politician's identity, rhetoric, promises, speaking style.
  Examples: "What is Pinarayi Vijayan's political identity?", "Analyze Yogi's rhetorical style"
- FACTUAL: Default — any other factual or general question about politics.
  Examples: "What did Mamata say about CAA?", "Latest news about farm protests"

IMPORTANT RULES:
1. Extract the politician's name EXACTLY as it would appear in the database (full name, properly capitalized).
   Known politicians include: Pinarayi Vijayan, Mamata Banerjee, Yogi Adityanath, and others.
2. Extract topic keywords (2-4 words capturing the subject).
3. Extract date range ONLY if explicitly mentioned (YYYY-MM format).
4. Extract source_leaning ONLY if the user mentions specific media orientation.

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "intent": "TEMPORAL_EVOLUTION" | "MEDIA_CONTRAST" | "PERSONA" | "FACTUAL",
  "politician": "Full Name" or "",
  "topic_keywords": ["keyword1", "keyword2"],
  "date_start": "YYYY-MM" or null,
  "date_end": "YYYY-MM" or null,
  "source_leaning": "left" | "center" | "right" or null
}}

USER QUERY: {query}
"""


class RAGOrchestrator:
    """
    Central orchestrator.

    Accepts optional llm_provider / api_key / model_name at init time
    to allow the Streamlit sidebar to override the .env defaults.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        logger.info("Initializing RAG Orchestrator (Dynamic Config)...")
        logger.info(f"  ENABLE_VECTOR_RAG = {settings.ENABLE_VECTOR_RAG}")
        logger.info(f"  ENABLE_GRAPH_RAG  = {settings.ENABLE_GRAPH_RAG}")
        logger.info(f"  LLM_PROVIDER      = {llm_provider or settings.LLM_PROVIDER}")

        if not settings.ENABLE_VECTOR_RAG and not settings.ENABLE_GRAPH_RAG:
            raise ConfigurationError(
                "At least one of ENABLE_VECTOR_RAG or ENABLE_GRAPH_RAG must be true."
            )

        self._vector_engine = None
        self._graph_engine = None
        self._generator = None

        if settings.ENABLE_VECTOR_RAG:
            self._vector_engine = self._init_vector_engine()

        if settings.ENABLE_GRAPH_RAG:
            self._graph_engine = self._init_graph_engine()

        # Generator — uses overrides if provided, else falls back to settings
        self._generator = self._init_generator(llm_provider, api_key, model_name)

        logger.info("RAG Orchestrator ready.")

    # ------------------------------------------------------------------
    # Initializers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_vector_engine():
        from src.rag.vector_engine import VectorRAGEngine
        return VectorRAGEngine()

    @staticmethod
    def _init_graph_engine():
        from src.rag.graph_engine import GraphRAGEngine
        return GraphRAGEngine()

    @staticmethod
    def _init_generator(
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        from src.rag.generator import Generator
        return Generator(
            llm_provider=llm_provider,
            api_key=api_key,
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_engines(self) -> List[str]:
        engines = []
        if self._vector_engine:
            engines.append("vector")
        if self._graph_engine:
            engines.append("graph")
        return engines

    @property
    def llm_backend(self) -> str:
        return self._generator.get_backend_name()

    @property
    def llm_model(self) -> str:
        return self._generator.get_model_name()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        engine: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a RAG query.

        Routing Logic:
          - engine="graph" -> Intent-Driven Pipeline
          - engine="vector" -> Legacy VectorRAG
          - engine=None     -> Prefer Graph if available
        """
        use_graph = False
        if engine == "graph" and self._graph_engine:
            use_graph = True
        elif engine == "vector" and self._vector_engine:
            use_graph = False
        elif engine is None:
            use_graph = self._graph_engine is not None

        if use_graph:
            return self._intent_pipeline(query, filters)
        else:
            return self._legacy_vector_pipeline(query, filters)

    # ------------------------------------------------------------------
    # INTENT-DRIVEN PIPELINE
    # ------------------------------------------------------------------

    def _intent_pipeline(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        # Step 1: Intent Classification
        analysis = self._classify_intent(query)
        logger.info(
            f"Intent: {analysis.intent.value} | "
            f"Politician: {analysis.politician} | "
            f"Topics: {analysis.topic_keywords}"
        )

        # Step 2: Vector Grounding
        grounded_ids = self._graph_engine.vector_ground(
            query, top_k=settings.TOP_K_RETRIEVAL
        )
        logger.info(f"Grounded {len(grounded_ids)} statement IDs.")

        if not grounded_ids:
            return self._empty_response(analysis.intent, "graph")

        # Step 3: Graph Traversal — route by intent
        context = self._execute_traversal(analysis, grounded_ids, query, filters)

        if context.total_statements == 0:
            return self._empty_response(analysis.intent, "graph")

        # Step 4: LLM Generation
        context.topic = " ".join(analysis.topic_keywords) if analysis.topic_keywords else query
        answer = self._generator.generate_intent_answer(query, context)

        sources = self._extract_sources(context)

        return {
            "answer": answer,
            "intent": analysis.intent.value,
            "analysis": {
                "politician": analysis.politician,
                "topic_keywords": analysis.topic_keywords,
                "date_start": analysis.date_start,
                "date_end": analysis.date_end,
                "source_leaning": analysis.source_leaning,
            },
            "sources": sources,
            "engine_used": "graph",
            "llm_backend": self.llm_backend,
        }

    def _classify_intent(self, query: str) -> QueryAnalysis:
        prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)

        try:
            raw_response = self._generator._call_llm(prompt)

            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            parsed = json.loads(cleaned)

            intent_str = parsed.get("intent", "FACTUAL").upper()
            try:
                intent = RAGIntent(intent_str)
            except ValueError:
                intent = RAGIntent.FACTUAL

            return QueryAnalysis(
                intent=intent,
                politician=parsed.get("politician", ""),
                topic_keywords=parsed.get("topic_keywords", []),
                date_start=parsed.get("date_start"),
                date_end=parsed.get("date_end"),
                source_leaning=parsed.get("source_leaning"),
            )

        except (json.JSONDecodeError, LLMGenerationError) as e:
            logger.warning(f"Intent classification failed, defaulting to FACTUAL: {e}")
            return QueryAnalysis(intent=RAGIntent.FACTUAL)

    def _execute_traversal(
        self,
        analysis: QueryAnalysis,
        grounded_ids: List[str],
        query: str,
        filters: Dict[str, Any] = None,
    ) -> RAGContext:
        intent = analysis.intent
        politician = analysis.politician

        if intent == RAGIntent.TEMPORAL_EVOLUTION and politician:
            return self._graph_engine.get_temporal_evolution(politician, grounded_ids)
        elif intent == RAGIntent.MEDIA_CONTRAST and politician:
            return self._graph_engine.get_media_portrayal(politician, grounded_ids)
        elif intent == RAGIntent.PERSONA and politician:
            return self._graph_engine.get_politician_persona(politician, grounded_ids)
        else:
            return self._graph_engine.get_factual_context(query, politician, filters)

    # ------------------------------------------------------------------
    # LEGACY VECTOR PIPELINE
    # ------------------------------------------------------------------

    def _legacy_vector_pipeline(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info("Querying with engine: vector")
        results = self._vector_engine.retrieve(query, filters=filters)

        if not results:
            return {
                "answer": "I couldn't find any information matching your query.",
                "sources": [],
                "engine_used": "vector",
                "llm_backend": self.llm_backend,
            }

        context = self._vector_engine.expand_context(results)
        answer = self._generator.generate_answer(query, context)

        return {
            "answer": answer,
            "sources": results,
            "engine_used": "vector",
            "llm_backend": self.llm_backend,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_response(self, intent: RAGIntent, engine: str) -> Dict[str, Any]:
        return {
            "answer": "I couldn't find any relevant information matching your query.",
            "intent": intent.value,
            "sources": [],
            "engine_used": engine,
            "llm_backend": self.llm_backend,
        }

    @staticmethod
    def _extract_sources(ctx: RAGContext) -> List[Dict[str, Any]]:
        sources = []

        if ctx.flat_statements:
            for stmt in ctx.flat_statements:
                sources.append({
                    "document": stmt.get("text", ""),
                    "metadata": {
                        "source": stmt.get("source", ""),
                        "politician": stmt.get("politician", ""),
                        "publish_date": stmt.get("publish_date", ""),
                        "title": stmt.get("title", ""),
                    },
                    "score": stmt.get("score", 0.0),
                })
        elif ctx.timeline:
            for entry in ctx.timeline:
                for stmt in entry.statements:
                    sources.append({
                        "document": stmt,
                        "metadata": {
                            "year_month": entry.year_month,
                            "sources": ", ".join(entry.sources),
                        },
                        "score": 0.0,
                    })
        elif ctx.media_groups:
            for group in ctx.media_groups:
                for stmt in group.statements:
                    sources.append({
                        "document": stmt,
                        "metadata": {
                            "leaning": group.leaning,
                            "sources": ", ".join(group.source_names),
                        },
                        "score": 0.0,
                    })
        elif ctx.rhetoric_statements or ctx.high_weight_statements:
            for stmt in ctx.rhetoric_statements + ctx.high_weight_statements:
                sources.append({
                    "document": stmt,
                    "metadata": {"type": "persona"},
                    "score": 0.0,
                })

        return sources
