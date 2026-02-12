import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List
from src.core.config import settings
from src.core.logger import logger
from src.models.schema import Statement

from chromadb.utils import embedding_functions

class VectorDBManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDBManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        logger.info(f"Initializing VectorDB at {settings.VECTOR_DB_DIR}")
        self.client = chromadb.PersistentClient(path=str(settings.VECTOR_DB_DIR))
        
        # Use the configured embedding model
        # ChromaDB requires an embedding function if we want it to handle embedding calculation
        # But wait, in upsert_statements, we are passing 'documents' (text).
        # Chroma calculates embeddings automatically using its default function (all-MiniLM-L6-v2) 
        # UNLESS we provide one.
        
        logger.info(f"Using Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        
        # Create or get collection with the custom embedding function
        self.collection = self.client.get_or_create_collection(
            name="politician_statements",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn
        )
        logger.info("VectorDB collection 'politician_statements' ready.")

    def upsert_statements(self, statements: List[Statement]):
        """
        Upserts a batch of statements into ChromaDB.
        """
        if not statements:
            return

        ids = []
        documents = []
        metadatas = []

        for stmt in statements:
            # Create a unique ID for the chunk: article_id + statement_index
            chunk_id = f"{stmt.article_id}_{stmt.statement_index}"
            
            ids.append(chunk_id)
            
            # Content to embed: Statement + Summary
            documents.append(f"{stmt.statement} {stmt.summary}")
            
            # Flatten metadata
            meta = {
                "article_id": stmt.article_id,
                "politician": stmt.politician,
                "classification": stmt.classification,
                "sentiment": stmt.sentiment,
                "publish_date": stmt.publish_date,
                "url": stmt.url,
                "source": stmt.source,
                "theme": ", ".join(stmt.theme), # Chroma doesn't support lists in metadata
                "temporal_focus": stmt.temporal_focus,
                "content_type": stmt.content_type,
                "perspective": stmt.perspective,
                # Store serialized entities if needed, or just key ones
                "entity_persons": ", ".join(stmt.article_entities.persons[:5]),
                "entity_orgs": ", ".join(stmt.article_entities.organizations[:5])
            }
            metadatas.append(meta)

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Upserted {len(statements)} statements.")
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")
            raise e
