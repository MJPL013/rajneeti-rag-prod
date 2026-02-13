"""
Generator — Intent-Driven LLM Synthesis.

Architecture:
  - Maintains the unified multi-provider abstraction (Gemini / Groq / Ollama).
  - Adds intent-specific prompt templates that produce structured,
    analytical answers instead of generic summaries.

Prompt Strategy Mapping:
  TEMPORAL_EVOLUTION  -> chronological stance trace
  MEDIA_CONTRAST      -> left vs center vs right comparison table
  PERSONA             -> rhetoric & promise analysis
  FACTUAL             -> standard Q&A with citations
"""

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import LLMGenerationError, ConfigurationError
from src.models.schema import RAGIntent, RAGContext

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


# ==================================================================
# INTENT -> PROMPT TEMPLATES
# ==================================================================

INTENT_PROMPTS = {
    RAGIntent.TEMPORAL_EVOLUTION: """You are a senior political analyst specializing in Indian politics.
You are given a CHRONOLOGICAL timeline of statements about the politician "{politician}" on the topic "{topic}".

TIMELINE DATA:
{context_block}

TASK:
1. Trace the EVOLUTION of {politician}'s stance or narrative on this topic over time.
2. Identify key turning points, shifts in rhetoric, or policy changes.
3. Note if external events (elections, crises) may have triggered shifts.
4. Use specific quotes from the data to support your analysis.

FORMAT: Write a structured chronological analysis with clear time markers.
If the data is sparse for some periods, explicitly note the gap.

Question: {query}
""",

    RAGIntent.MEDIA_CONTRAST: """You are a media analysis expert specializing in Indian political journalism.
You are given statements about the politician "{politician}" GROUPED BY MEDIA LEANING.

MEDIA COVERAGE DATA:
{context_block}

TASK:
1. Compare how LEFT-leaning, CENTER, and RIGHT-leaning media portray {politician}.
2. Identify framing differences: tone, emphasis, omissions.
3. Note which aspects each media group highlights or downplays.
4. Be balanced — present observations without personal bias.

FORMAT: Use a structured comparison (e.g., Left says X, Center says Y, Right says Z).
Cite the specific source names (e.g., TheWire, NDTV, ThePrint).

Question: {query}
""",

    RAGIntent.PERSONA: """You are a political profiling expert analyzing Indian politicians.
You are given HIGH-WEIGHT RHETORIC and key statements by or about "{politician}".

RHETORIC & KEY STATEMENTS:
{context_block}

TASK:
1. Analyze the political PERSONA of {politician}.
2. Identify their core themes, promises, rhetorical style.
3. Note recurring patterns in how they communicate.
4. Distinguish between their own words (direct quotes) vs. media characterizations.

FORMAT: Write a political profile with sections for:
- Core Messaging & Themes
- Rhetorical Style
- Key Policy Positions (if evident)
- Notable Quotes

Question: {query}
""",

    RAGIntent.FACTUAL: """You are a helpful political analyst assistant.
Use the provided context to answer the user's question.
The context consists of news articles and statements about politicians.

CONTEXT:
{context_block}

Rules:
1. Answer ONLY based on the provided context.
2. If the answer is not in the context, say "I don't have enough information."
3. Cite the politician or source if mentioned in the context.
4. Be objective and factual.

Question: {query}
""",
}


# ==================================================================
# CONTEXT FORMATTERS (RAGContext -> str for prompt)
# ==================================================================

def _format_temporal_context(ctx: RAGContext) -> str:
    """Format timeline entries into a readable block."""
    lines = []
    for entry in ctx.timeline:
        lines.append(f"\n=== {entry.year_month} ===")
        lines.append(f"  Sources: {', '.join(entry.sources) if entry.sources else 'N/A'}")
        for i, stmt in enumerate(entry.statements, 1):
            lines.append(f"  [{i}] {stmt}")
    return "\n".join(lines) if lines else "No timeline data available."


def _format_media_context(ctx: RAGContext) -> str:
    """Format media groups into a readable block."""
    lines = []
    for group in ctx.media_groups:
        lines.append(f"\n--- {group.leaning.upper()} MEDIA ({', '.join(group.source_names)}) ---")
        for i, stmt in enumerate(group.statements, 1):
            lines.append(f"  [{i}] {stmt}")
    return "\n".join(lines) if lines else "No media comparison data available."


def _format_persona_context(ctx: RAGContext) -> str:
    """Format persona statements into a readable block."""
    lines = []
    if ctx.rhetoric_statements:
        lines.append("\n--- RHETORIC (Politician's Own Voice) ---")
        for i, stmt in enumerate(ctx.rhetoric_statements, 1):
            lines.append(f"  [{i}] {stmt}")

    if ctx.high_weight_statements:
        lines.append("\n--- HIGH-WEIGHT STATEMENTS (Headlines / Major) ---")
        for i, stmt in enumerate(ctx.high_weight_statements, 1):
            lines.append(f"  [{i}] {stmt}")

    return "\n".join(lines) if lines else "No persona data available."


def _format_factual_context(ctx: RAGContext) -> str:
    """Format flat statement list into a readable block."""
    lines = []
    for i, stmt in enumerate(ctx.flat_statements, 1):
        lines.append(
            f"\n[{i}] {stmt.get('text', '')}"
            f"\n    Source: {stmt.get('source', 'N/A')} | "
            f"Date: {stmt.get('publish_date', 'N/A')} | "
            f"Politician: {stmt.get('politician', 'N/A')}"
        )
    return "\n".join(lines) if lines else "No relevant information found."


CONTEXT_FORMATTERS = {
    RAGIntent.TEMPORAL_EVOLUTION: _format_temporal_context,
    RAGIntent.MEDIA_CONTRAST: _format_media_context,
    RAGIntent.PERSONA: _format_persona_context,
    RAGIntent.FACTUAL: _format_factual_context,
}


# ==================================================================
# GENERATOR CLASS
# ==================================================================

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
    # Provider init methods
    # ------------------------------------------------------------------

    def _init_gemini(self):
        if not settings.GEMINI_API_KEY:
            raise ConfigurationError("GEMINI_API_KEY is required when LLM_PROVIDER='gemini'.")

        genai = _get_genai()
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
            logger.info(f"Gemini initialized -- model: {settings.GEMINI_MODEL_NAME}")
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
            logger.info(f"Groq initialized -- model: {settings.GROQ_MODEL_NAME}")
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
            logger.info(f"Ollama initialized -- model: {settings.LLM_MODEL_NAME} @ {settings.OLLAMA_BASE_URL}")
        except Exception as e:
            raise LLMGenerationError(f"Failed to initialize Ollama: {e}") from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_backend_name(self) -> str:
        return self.backend

    def generate_answer(self, query: str, context: str) -> str:
        """
        Legacy API — plain string context (backwards-compatible).
        """
        if not context:
            return "I could not find any relevant information to answer your question."

        prompt = INTENT_PROMPTS[RAGIntent.FACTUAL].format(
            context_block=context,
            query=query,
            politician="",
            topic="",
        )
        return self._call_llm(prompt)

    def generate_intent_answer(self, query: str, context: RAGContext) -> str:
        """
        New API — accepts structured RAGContext, formats prompt by intent.
        """
        if context.total_statements == 0:
            return "I could not find enough relevant information in the graph to answer your question."

        intent = context.intent
        formatter = CONTEXT_FORMATTERS.get(intent, _format_factual_context)
        context_block = formatter(context)

        template = INTENT_PROMPTS.get(intent, INTENT_PROMPTS[RAGIntent.FACTUAL])
        prompt = template.format(
            context_block=context_block,
            query=query,
            politician=context.politician or "the politician",
            topic=context.topic or "the topic",
        )

        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Unified LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Route to the active backend."""
        if self.backend == "gemini":
            return self._generate_gemini(prompt)
        elif self.backend == "groq":
            return self._generate_groq(prompt)
        elif self.backend == "ollama":
            return self._generate_ollama(prompt)
        else:
            raise LLMGenerationError(f"No LLM backend configured (provider={self.backend}).")

    def _generate_gemini(self, prompt: str) -> str:
        try:
            logger.info("Sending request to Gemini API...")
            response = self._gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise LLMGenerationError(f"Gemini generation failed: {e}") from e

    def _generate_groq(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            logger.info("Sending request to Groq...")
            messages = [HumanMessage(content=prompt)]
            response = self._groq_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise LLMGenerationError(f"Groq generation failed: {e}") from e

    def _generate_ollama(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            logger.info("Sending request to Ollama...")
            messages = [HumanMessage(content=prompt)]
            response = self._ollama_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise LLMGenerationError(f"Ollama generation failed: {e}") from e
