"""
Rajneeti Schema — Pydantic models for ingestion AND retrieval.

Ingestion Models : ArticleEntities, Statement, NewsArticle
RAG Models       : RAGIntent, QueryAnalysis, RAGContext, RAGResponse
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================
# INGESTION MODELS (Unchanged)
# ============================================================

class ArticleEntities(BaseModel):
    persons: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    policies_schemes: List[str] = Field(default_factory=list)


class Statement(BaseModel):
    # Core Content
    statement: str
    summary: str
    weight: float

    # Metadata (Statement Level)
    theme: List[str] = Field(default_factory=list)
    classification: str
    temporal_focus: str
    content_type: str
    perspective: str
    sentiment: str

    # Metadata (Article Level - Enriched)
    article_id: str
    statement_index: int
    politician: str
    source: str
    publish_date: Optional[str] = None
    url: str
    title: str
    is_relevant: bool = True
    article_entities: ArticleEntities


class NewsArticle(BaseModel):
    article_id: str
    politician: str
    source: str
    publish_date: Optional[str] = None
    url: str
    title: str
    relevance_level: str
    article_entities: ArticleEntities
    statements: List[dict]  # Raw dicts to be processed into Statement objects


# ============================================================
# RAG MODELS (New — Intent-Driven Pipeline)
# ============================================================

class RAGIntent(str, Enum):
    """
    Classifies the user's query intent to route to the correct
    graph traversal strategy.

    TEMPORAL_EVOLUTION : "How has X's stance on Y changed over time?"
    MEDIA_CONTRAST     : "How do left vs right media portray X?"
    PERSONA            : "What is X's political identity / rhetoric?"
    FACTUAL            : General / factual lookup (default fallback)
    """
    TEMPORAL_EVOLUTION = "TEMPORAL_EVOLUTION"
    MEDIA_CONTRAST = "MEDIA_CONTRAST"
    PERSONA = "PERSONA"
    FACTUAL = "FACTUAL"


class QueryAnalysis(BaseModel):
    """
    Output of the LLM-based intent classifier.
    Extracts structured signals from a natural-language query.
    """
    intent: RAGIntent = RAGIntent.FACTUAL
    politician: str = ""
    topic_keywords: List[str] = Field(default_factory=list)
    date_start: Optional[str] = None     # YYYY-MM format
    date_end: Optional[str] = None       # YYYY-MM format
    source_leaning: Optional[str] = None  # "left" | "center" | "right"


class TimelineEntry(BaseModel):
    """A single point on the temporal evolution timeline."""
    year_month: str
    statements: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class MediaGroup(BaseModel):
    """Statements grouped by media political leaning."""
    leaning: str  # "left" | "center" | "center-left" | "center-right" | "right"
    source_names: List[str] = Field(default_factory=list)
    statements: List[str] = Field(default_factory=list)


class RAGContext(BaseModel):
    """
    Structured context passed from GraphEngine to Generator.
    Shape depends on the intent — only populate the relevant field.
    """
    intent: RAGIntent = RAGIntent.FACTUAL
    politician: str = ""
    topic: str = ""

    # TEMPORAL_EVOLUTION
    timeline: List[TimelineEntry] = Field(default_factory=list)

    # MEDIA_CONTRAST
    media_groups: List[MediaGroup] = Field(default_factory=list)

    # PERSONA
    rhetoric_statements: List[str] = Field(default_factory=list)
    high_weight_statements: List[str] = Field(default_factory=list)

    # FACTUAL (flat list fallback)
    flat_statements: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    total_statements: int = 0
    sources_used: List[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Final structured output returned to the frontend / API."""
    answer: str
    intent: RAGIntent = RAGIntent.FACTUAL
    context_used: Optional[RAGContext] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    engine_used: str = "graph"
    llm_backend: str = ""
