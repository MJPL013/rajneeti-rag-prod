"""
Generator module — Unified LLM abstraction layer.
The ONLY entry point for LLM calls in the application.
Provider is selected via settings.LLM_PROVIDER ("gemini" | "groq" | "ollama").
"""
from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import LLMGenerationError, ConfigurationError

# --- Lazy imports for optional providers ---
_genai = None
_ChatGroq = None
_ChatOllama = None


def _get_genai():
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            _genai = genai
        except ImportError:
            raise ConfigurationError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            )
    return _genai


def _get_chat_groq():
    global _ChatGroq
    if _ChatGroq is None:
        try:
            from langchain_groq import ChatGroq
            _ChatGroq = ChatGroq
        except ImportError:
            raise ConfigurationError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            )
    return _ChatGroq


def _get_chat_ollama():
    global _ChatOllama
    if _ChatOllama is None:
        try:
            from langchain_community.chat_models import ChatOllama
            _ChatOllama = ChatOllama
        except ImportError:
            raise ConfigurationError(
                "langchain-community is not installed. Run: pip install langchain-community"
            )
    return _ChatOllama


class Generator:
    """
    Singleton LLM Generator.
    Backend is determined exclusively by settings.LLM_PROVIDER.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Generator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        provider = settings.LLM_PROVIDER.lower().strip()
        logger.info(f"LLM Provider: {provider}")

        self.backend = provider
        self._ollama_llm = None
        self._gemini_model = None
        self._groq_llm = None

        if provider == "gemini":
            self._init_gemini()
        elif provider == "groq":
            self._init_groq()
        elif provider == "ollama":
            self._init_ollama()
        else:
            raise ConfigurationError(
                f"Unknown LLM_PROVIDER '{provider}'. Must be 'gemini', 'groq', or 'ollama'."
            )

    # ------------------------------------------------------------------
    # Provider init methods — each raises on failure (no silent fallback)
    # ------------------------------------------------------------------

    def _init_gemini(self):
        if not settings.GEMINI_API_KEY:
            raise ConfigurationError("GEMINI_API_KEY is required when LLM_PROVIDER='gemini'.")

        genai = _get_genai()
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
            logger.info(f"Gemini initialized — model: {settings.GEMINI_MODEL_NAME}")
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize Gemini: {e}") from e

    def _init_groq(self):
        if not settings.GROQ_API_KEY:
            raise ConfigurationError("GROQ_API_KEY is required when LLM_PROVIDER='groq'.")

        ChatGroq = _get_chat_groq()
        try:
            self._groq_llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL_NAME,
                temperature=0.3,
            )
            logger.info(f"Groq initialized — model: {settings.GROQ_MODEL_NAME}")
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize Groq: {e}") from e

    def _init_ollama(self):
        ChatOllama = _get_chat_ollama()
        try:
            self._ollama_llm = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.LLM_MODEL_NAME,
                temperature=0.3,
            )
            logger.info(f"Ollama initialized — model: {settings.LLM_MODEL_NAME} @ {settings.OLLAMA_BASE_URL}")
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize Ollama: {e}") from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_backend_name(self) -> str:
        """Returns the active backend name."""
        return self.backend

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using the configured LLM backend."""
        if not context:
            return "I could not find any relevant information to answer your question."

        system_prompt = """You are a helpful political analyst assistant. 
        Use the provided context to answer the user's question. 
        The context consists of news articles and statements about politicians.
        
        Rules:
        1. Answer ONLY based on the provided context.
        2. If the answer is not in the context, say "I don't have enough information."
        3. Cite the politician or source if mentioned in the context.
        4. Be objective and factual.
        """

        user_message = f"""Context:
        {context}
        
        Question: {query}
        """

        if self.backend == "gemini":
            return self._generate_gemini(system_prompt, user_message)
        elif self.backend == "groq":
            return self._generate_groq(system_prompt, user_message)
        elif self.backend == "ollama":
            return self._generate_ollama(system_prompt, user_message)
        else:
            raise LLMGenerationError(f"No LLM backend configured (provider={self.backend}).")

    # ------------------------------------------------------------------
    # Private generation methods
    # ------------------------------------------------------------------

    def _generate_gemini(self, system_prompt: str, user_message: str) -> str:
        try:
            logger.info("Sending request to Gemini API...")
            full_prompt = f"{system_prompt}\n\n{user_message}"
            response = self._gemini_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise LLMGenerationError(f"Gemini generation failed: {e}") from e

    def _generate_groq(self, system_prompt: str, user_message: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            logger.info("Sending request to Groq...")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self._groq_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise LLMGenerationError(f"Groq generation failed: {e}") from e

    def _generate_ollama(self, system_prompt: str, user_message: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            logger.info("Sending request to Ollama...")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self._ollama_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise LLMGenerationError(f"Ollama generation failed: {e}") from e
