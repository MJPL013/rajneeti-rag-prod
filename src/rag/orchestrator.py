"""
RAG Orchestrator — single entry point for all RAG operations.

Responsibilities:
  1. Accept query + filters
  2. Route to VectorRAG and/or GraphRAG based on feature flags
  3. Merge results from multiple engines
  4. Expand context from the active engine
  5. Call Generator for answer
  6. Return structured response
"""
from typing import Dict, Any, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import ConfigurationError


class RAGOrchestrator:
    """
    Central orchestrator that coordinates retrieval engines and the generator.
    Engines are initialized lazily based on feature flags.
    """

    def __init__(self):
        logger.info("Initializing RAG Orchestrator...")
        logger.info(f"  ENABLE_VECTOR_RAG = {settings.ENABLE_VECTOR_RAG}")
        logger.info(f"  ENABLE_GRAPH_RAG  = {settings.ENABLE_GRAPH_RAG}")
        logger.info(f"  LLM_PROVIDER      = {settings.LLM_PROVIDER}")

        if not settings.ENABLE_VECTOR_RAG and not settings.ENABLE_GRAPH_RAG:
            raise ConfigurationError(
                "At least one of ENABLE_VECTOR_RAG or ENABLE_GRAPH_RAG must be true."
            )

        self._vector_engine = None
        self._graph_engine = None
        self._generator = None

        # Eagerly init enabled engines so startup errors surface immediately
        if settings.ENABLE_VECTOR_RAG:
            self._vector_engine = self._init_vector_engine()

        if settings.ENABLE_GRAPH_RAG:
            self._graph_engine = self._init_graph_engine()

        # Generator (singleton — safe to call multiple times)
        self._generator = self._init_generator()

        logger.info("RAG Orchestrator ready.")

    # ------------------------------------------------------------------
    # Lazy initializers
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
    def _init_generator():
        from src.rag.generator import Generator
        return Generator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_engines(self) -> List[str]:
        """Returns list of enabled engine names."""
        engines = []
        if self._vector_engine:
            engines.append("vector")
        if self._graph_engine:
            engines.append("graph")
        return engines

    @property
    def llm_backend(self) -> str:
        return self._generator.get_backend_name()

    def query(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        engine: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a RAG query.

        Args:
            query: User question
            filters: Optional metadata filters (classification, etc.)
            engine: Force a specific engine ("vector" or "graph").
                    If None, uses the first available engine.

        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "engine_used": str,
                "llm_backend": str,
            }
        """
        # Decide which engine to use
        active_engine, engine_name = self._select_engine(engine)

        # Retrieve
        logger.info(f"Querying with engine: {engine_name}")
        results = active_engine.retrieve(query, filters=filters)

        if not results:
            return {
                "answer": "I couldn't find any information matching your query.",
                "sources": [],
                "engine_used": engine_name,
                "llm_backend": self.llm_backend,
            }

        # Expand context
        context = active_engine.expand_context(results)

        # Generate answer
        answer = self._generator.generate_answer(query, context)

        return {
            "answer": answer,
            "sources": results,
            "engine_used": engine_name,
            "llm_backend": self.llm_backend,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_engine(self, preferred: Optional[str] = None):
        """Returns (engine_instance, engine_name) based on preference or availability."""
        if preferred == "graph" and self._graph_engine:
            return self._graph_engine, "graph"
        if preferred == "vector" and self._vector_engine:
            return self._vector_engine, "vector"

        # Default: prefer vector if available, else graph
        if self._vector_engine:
            return self._vector_engine, "vector"
        if self._graph_engine:
            return self._graph_engine, "graph"

        raise ConfigurationError("No RAG engine is available.")
