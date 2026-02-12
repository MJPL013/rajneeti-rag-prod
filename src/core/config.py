from pydantic_settings import BaseSettings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # --- Environment ---
    APP_ENV: str = "local"  # "local" | "aws"

    # --- Project Paths (overridden by env vars for absolute paths) ---
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    RAW_DATA_DIR: Path = BASE_DIR / "raw_data"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    
    # --- Embedding & Reranking ---
    EMBEDDING_MODEL_NAME: str = "google/embeddinggemma-300m"
    RERANKING_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM Provider Selection ---
    LLM_PROVIDER: str = "gemini"  # "gemini" | "groq" | "ollama"

    # --- Gemini ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-flash-latest"

    # --- Groq ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "openai/gpt-oss-120b"

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = "llama3.2"  # Ollama model name

    # --- Feature Flags ---
    ENABLE_GRAPH_RAG: bool = False
    ENABLE_VECTOR_RAG: bool = True

    # --- Graph Database (Neo4j Aura) ---
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"

    # --- Retrieval ---
    TOP_K_RETRIEVAL: int = 7
    TOP_K_RERANK: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
