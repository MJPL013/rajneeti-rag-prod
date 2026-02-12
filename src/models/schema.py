from typing import List, Optional
from pydantic import BaseModel, Field

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
    statements: List[dict] # Raw dicts to be processed into Statement objects
