"""
Generator — Intent-Driven LLM Synthesis (Multi-Provider).

Supports dynamic provider switching at runtime:
  Gemini | Groq | OpenAI | DeepSeek | ZhipuAI (GLM)

No longer a singleton — each instance owns its own LLM connection,
allowing the Streamlit frontend to swap providers via session state.
"""

from typing import Optional

from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import LLMGenerationError, ConfigurationError
from src.models.schema import RAGIntent, RAGContext


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
    lines = []
    for entry in ctx.timeline:
        lines.append(f"\n=== {entry.year_month} ===")
        lines.append(f"  Sources: {', '.join(entry.sources) if entry.sources else 'N/A'}")
        for i, stmt in enumerate(entry.statements, 1):
            lines.append(f"  [{i}] {stmt}")
    return "\n".join(lines) if lines else "No timeline data available."


def _format_media_context(ctx: RAGContext) -> str:
    lines = []
    for group in ctx.media_groups:
        lines.append(f"\n--- {group.leaning.upper()} MEDIA ({', '.join(group.source_names)}) ---")
        for i, stmt in enumerate(group.statements, 1):
            lines.append(f"  [{i}] {stmt}")
    return "\n".join(lines) if lines else "No media comparison data available."


def _format_persona_context(ctx: RAGContext) -> str:
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
# PROVIDER REGISTRY — maps provider key to (init_fn_name, generate_fn_name)
# ==================================================================

# Provider keys used in the sidebar dropdown and config
SUPPORTED_PROVIDERS = [
    "gemini",
    "groq",
    "openai",
    "deepseek",
    "zhipuai",
]

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "zhipuai": "glm-4-flash",
}


# ==================================================================
# GENERATOR CLASS (No longer singleton)
# ==================================================================

class Generator:
    """
    LLM Generator — one instance per provider configuration.
    Accepts runtime overrides for provider, api_key, and model_name.
    Falls back to settings.* defaults when overrides are not provided.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        # Resolve provider
        self.backend = (llm_provider or settings.LLM_PROVIDER).lower().strip()

        # Resolve API key (provider-specific fallback from settings)
        self.api_key = api_key or self._default_api_key(self.backend)

        # Resolve model name
        self.model_name = model_name or self._default_model_name(self.backend)

        logger.info(f"Generator init: provider={self.backend}, model={self.model_name}")

        # LLM handles (only the active one will be set)
        self._gemini_model = None
        self._groq_llm = None
        self._openai_client = None
        self._deepseek_client = None
        self._zhipuai_client = None

        # Initialize the chosen provider
        init_map = {
            "gemini": self._init_gemini,
            "groq": self._init_groq,
            "openai": self._init_openai,
            "deepseek": self._init_deepseek,
            "zhipuai": self._init_zhipuai,
        }

        init_fn = init_map.get(self.backend)
        if init_fn is None:
            raise ConfigurationError(
                f"Unknown LLM provider '{self.backend}'. "
                f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
            )
        init_fn()

    # ------------------------------------------------------------------
    # Default resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_api_key(provider: str) -> str:
        """Pull the matching API key from settings / .env."""
        key_map = {
            "gemini": settings.GEMINI_API_KEY,
            "groq": settings.GROQ_API_KEY,
            "openai": getattr(settings, "OPENAI_API_KEY", ""),
            "deepseek": getattr(settings, "DEEPSEEK_API_KEY", ""),
            "zhipuai": getattr(settings, "ZHIPUAI_API_KEY", ""),
        }
        return key_map.get(provider, "")

    @staticmethod
    def _default_model_name(provider: str) -> str:
        """Pull the matching model name from settings, else use DEFAULT_MODELS."""
        model_map = {
            "gemini": settings.GEMINI_MODEL_NAME,
            "groq": settings.GROQ_MODEL_NAME,
        }
        return model_map.get(provider, DEFAULT_MODELS.get(provider, ""))

    # ------------------------------------------------------------------
    # Provider init methods
    # ------------------------------------------------------------------

    def _init_gemini(self):
        if not self.api_key:
            raise ConfigurationError("API key is required for Gemini.")
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._gemini_model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini ready -> {self.model_name}")
        except ImportError:
            raise ConfigurationError("pip install google-generativeai")
        except Exception as e:
            raise LLMGenerationError(f"Gemini init failed: {e}") from e

    def _init_groq(self):
        if not self.api_key:
            raise ConfigurationError("API key is required for Groq.")
        try:
            from langchain_groq import ChatGroq
            self._groq_llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.3,
            )
            logger.info(f"Groq ready -> {self.model_name}")
        except ImportError:
            raise ConfigurationError("pip install langchain-groq")
        except Exception as e:
            raise LLMGenerationError(f"Groq init failed: {e}") from e

    def _init_openai(self):
        if not self.api_key:
            raise ConfigurationError("API key is required for OpenAI.")
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI ready -> {self.model_name}")
        except ImportError:
            raise ConfigurationError("pip install openai")
        except Exception as e:
            raise LLMGenerationError(f"OpenAI init failed: {e}") from e

    def _init_deepseek(self):
        if not self.api_key:
            raise ConfigurationError("API key is required for DeepSeek.")
        try:
            from openai import OpenAI
            self._deepseek_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )
            logger.info(f"DeepSeek ready -> {self.model_name}")
        except ImportError:
            raise ConfigurationError("pip install openai  (DeepSeek uses OpenAI-compatible API)")
        except Exception as e:
            raise LLMGenerationError(f"DeepSeek init failed: {e}") from e

    def _init_zhipuai(self):
        if not self.api_key:
            raise ConfigurationError("API key is required for ZhipuAI/GLM.")
        try:
            from openai import OpenAI
            self._zhipuai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4",
            )
            logger.info(f"ZhipuAI/GLM ready -> {self.model_name}")
        except ImportError:
            raise ConfigurationError("pip install openai  (ZhipuAI uses OpenAI-compatible API)")
        except Exception as e:
            raise LLMGenerationError(f"ZhipuAI init failed: {e}") from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_backend_name(self) -> str:
        return self.backend

    def get_model_name(self) -> str:
        return self.model_name

    def generate_answer(self, query: str, context: str) -> str:
        """Legacy API — plain string context."""
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
        """New API — accepts structured RAGContext, formats prompt by intent."""
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
    # Unified LLM call — routes to the active backend
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        dispatch = {
            "gemini": self._generate_gemini,
            "groq": self._generate_groq,
            "openai": self._generate_openai,
            "deepseek": self._generate_deepseek,
            "zhipuai": self._generate_zhipuai,
        }
        fn = dispatch.get(self.backend)
        if fn is None:
            raise LLMGenerationError(f"No LLM backend configured (provider={self.backend}).")
        return fn(prompt)

    # ------------------------------------------------------------------
    # Provider-specific generation
    # ------------------------------------------------------------------

    def _generate_gemini(self, prompt: str) -> str:
        try:
            response = self._gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise LLMGenerationError(f"Gemini generation failed: {e}") from e

    def _generate_groq(self, prompt: str) -> str:
        try:
            from langchain_core.messages import HumanMessage
            response = self._groq_llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise LLMGenerationError(f"Groq generation failed: {e}") from e

    def _generate_openai(self, prompt: str) -> str:
        try:
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise LLMGenerationError(f"OpenAI generation failed: {e}") from e

    def _generate_deepseek(self, prompt: str) -> str:
        try:
            response = self._deepseek_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            raise LLMGenerationError(f"DeepSeek generation failed: {e}") from e

    def _generate_zhipuai(self, prompt: str) -> str:
        try:
            response = self._zhipuai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"ZhipuAI generation failed: {e}")
            raise LLMGenerationError(f"ZhipuAI generation failed: {e}") from e
