class RAGException(Exception):
    """Base exception for the RAG application."""
    pass

class ConfigurationError(RAGException):
    """Raised when a required configuration is missing or invalid."""
    pass

class DataIngestionError(RAGException):
    """Raised when data ingestion fails."""
    pass

class VectorDBError(RAGException):
    """Raised when vector database operations fail."""
    pass

class GraphDBError(RAGException):
    """Raised when graph database operations fail."""
    pass

class RetrievalError(RAGException):
    """Raised when retrieval fails."""
    pass

class LLMGenerationError(RAGException):
    """Raised when LLM generation fails."""
    pass
