"""
Rajneeti RAG — Streamlit Frontend (Intent-Driven, Dynamic Config).

Features:
  1. Conversational guardrail — greetings & meta-questions bypass RAG
  2. Suggested query presets — showcases temporal, media, persona, factual
  3. Dynamic sidebar — switch LLM provider / API key / model at runtime
  4. Intent badges & analysis display on GraphRAG responses
"""

import re
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.core.config import settings

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Rajneeti — Political Intelligence Engine",
    page_icon="🏛️",
    layout="wide",
)


# ==================================================================
# CONVERSATIONAL GUARDRAIL
# ==================================================================

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hola|namaste|good\s?(morning|evening|afternoon)|sup|yo)\b",
    re.IGNORECASE,
)

_HELP_PATTERNS = re.compile(
    r"(what (is|are) (this|you|rajneeti)|help|what can you do|how (do|does) (this|it) work)",
    re.IGNORECASE,
)

_THANKS_PATTERNS = re.compile(
    r"^(thanks?|thank\s?you|thx|ty|cheers)\b",
    re.IGNORECASE,
)

GREETING_RESPONSE = (
    "Hello! 👋 I'm **Rajneeti**, your Political Intelligence Engine.\n\n"
    "Ask me deep analytical questions about Indian politicians — "
    "I can trace stance evolution over time, compare media portrayals, "
    "analyze political personas, and answer factual queries.\n\n"
    "Try one of the **suggested queries** below to get started!"
)

HELP_RESPONSE = (
    "**Rajneeti** is an AI-powered political analysis system built on a knowledge graph "
    "of Indian political news.\n\n"
    "**What I can do:**\n"
    "- 🕐 **Temporal Analysis** — Trace how a politician's stance has evolved over time\n"
    "- 📰 **Media Contrast** — Compare left, center, and right-leaning media coverage\n"
    "- 🎭 **Persona Profiling** — Analyze a politician's rhetoric, promises & style\n"
    "- 🔍 **Factual Q&A** — Answer specific questions with cited sources\n\n"
    "Use the **sidebar** to switch LLM providers or change the backend engine."
)

THANKS_RESPONSE = "You're welcome! Feel free to ask another question anytime. 🙏"


def check_conversational(text: str):
    text = text.strip()
    if _GREETING_PATTERNS.search(text):
        return GREETING_RESPONSE
    if _HELP_PATTERNS.search(text):
        return HELP_RESPONSE
    if _THANKS_PATTERNS.search(text):
        return THANKS_RESPONSE
    return None


# ==================================================================
# QUERY PRESETS
# ==================================================================

QUERY_PRESETS = {
    "🕐 Temporal": "Trace the evolution of Yogi Adityanath's stance on infrastructure development since 2019.",
    "📰 Media Contrast": "Compare how Left-leaning and Center-leaning media portray Mamata Banerjee on the CAA issue.",
    "🎭 Persona": "Analyze the core rhetoric, promises, and political identity of Pinarayi Vijayan regarding welfare policies.",
    "🔍 Factual": "What has Mamata Banerjee said about the Citizenship Amendment Act?",
}


# ==================================================================
# PROVIDER CONFIG
# ==================================================================

PROVIDER_DISPLAY = {
    "gemini": "Gemini",
    "groq": "Groq",
    "openai": "OpenAI (ChatGPT)",
    "deepseek": "DeepSeek",
    "zhipuai": "GLM / ZhipuAI",
}

PROVIDER_KEYS = {v: k for k, v in PROVIDER_DISPLAY.items()}

DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "zhipuai": "glm-4-flash",
}


def _default_api_key_for(provider: str) -> str:
    return {
        "gemini": settings.GEMINI_API_KEY,
        "groq": settings.GROQ_API_KEY,
        "openai": getattr(settings, "OPENAI_API_KEY", ""),
        "deepseek": getattr(settings, "DEEPSEEK_API_KEY", ""),
        "zhipuai": getattr(settings, "ZHIPUAI_API_KEY", ""),
    }.get(provider, "")


def _default_model_for(provider: str) -> str:
    return {
        "gemini": settings.GEMINI_MODEL_NAME,
        "groq": settings.GROQ_MODEL_NAME,
        "openai": getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini"),
        "deepseek": getattr(settings, "DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        "zhipuai": getattr(settings, "ZHIPUAI_MODEL_NAME", "glm-4-flash"),
    }.get(provider, "")


# ==================================================================
# ORCHESTRATOR FACTORY (keyed cache)
# ==================================================================

@st.cache_resource
def get_orchestrator(provider: str, api_key: str, model_name: str):
    from src.rag.orchestrator import RAGOrchestrator
    return RAGOrchestrator(
        llm_provider=provider,
        api_key=api_key,
        model_name=model_name,
    )


# ==================================================================
# MAIN
# ==================================================================

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = settings.LLM_PROVIDER.lower().strip()
    if "api_key" not in st.session_state:
        st.session_state.api_key = _default_api_key_for(st.session_state.llm_provider)
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODELS.get(
            st.session_state.llm_provider,
            settings.GEMINI_MODEL_NAME,
        )

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("LLM Provider")

        provider_names = list(PROVIDER_DISPLAY.values())
        current_display = PROVIDER_DISPLAY.get(st.session_state.llm_provider, "Gemini")
        current_idx = provider_names.index(current_display) if current_display in provider_names else 0

        selected_display = st.selectbox("Provider", provider_names, index=current_idx)
        selected_provider = PROVIDER_KEYS[selected_display]

        if selected_provider != st.session_state.llm_provider:
            st.session_state.llm_provider = selected_provider
            st.session_state.api_key = _default_api_key_for(selected_provider)
            st.session_state.model_name = _default_model_for(selected_provider)
            st.rerun()

        st.divider()

        st.subheader("Retrieval Engine")
        engine_options = []
        engine_keys = []
        if settings.ENABLE_GRAPH_RAG:
            engine_options.append("GraphRAG (Neo4j)")
            engine_keys.append("graph")
        if settings.ENABLE_VECTOR_RAG:
            engine_options.append("ChromaDB (Vector)")
            engine_keys.append("vector")

        if len(engine_options) > 1:
            selected_idx = st.radio("Backend", range(len(engine_options)), format_func=lambda i: engine_options[i])
            selected_engine = engine_keys[selected_idx]
        elif engine_options:
            st.info(f"Engine: {engine_options[0]}")
            selected_engine = engine_keys[0]
        else:
            st.error("No retrieval engine enabled!")
            st.stop()

        st.divider()
        st.subheader("Status")
        st.markdown(f"**Provider:** `{PROVIDER_DISPLAY.get(st.session_state.llm_provider, st.session_state.llm_provider)}`")
        st.markdown(f"**Model:** `{st.session_state.model_name}`")
        st.markdown(f"**Engine:** `{selected_engine}`")
        st.markdown(f"**Environment:** `{settings.APP_ENV}`")
        st.info("💡 Credentials are loaded from .env")

    try:
        orchestrator = get_orchestrator(
            st.session_state.llm_provider,
            st.session_state.api_key,
            st.session_state.model_name,
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()

    st.title("🏛️ Rajneeti")
    st.caption("Political Intelligence Engine — Intent-Driven Graph RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("##### 💡 Try a deep query")
    preset_cols = st.columns(len(QUERY_PRESETS))
    for i, (label, query_text) in enumerate(QUERY_PRESETS.items()):
        with preset_cols[i]:
            if st.button(label, use_container_width=True, key=f"preset_{i}"):
                st.session_state.pending_query = query_text
                st.rerun()

    user_input = st.chat_input("Ask a question about Indian politics...")
    prompt = st.session_state.pending_query or user_input
    st.session_state.pending_query = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        guardrail_response = check_conversational(prompt)

        with st.chat_message("assistant"):
            if guardrail_response:
                st.markdown(guardrail_response)
                answer = guardrail_response
            else:
                provider_label = PROVIDER_DISPLAY.get(st.session_state.llm_provider, st.session_state.llm_provider)
                st.caption(f"LLM: {provider_label} ({st.session_state.model_name}) | Engine: {selected_engine}")

                with st.spinner("Classifying intent & retrieving context..."):
                    result = orchestrator.query(query=prompt, filters={}, engine=selected_engine)

                    sources = result.get("sources", [])
                    answer = result["answer"]
                    intent = result.get("intent")
                    analysis = result.get("analysis")

                    if intent:
                        intent_colors = {
                            "TEMPORAL_EVOLUTION": "blue",
                            "MEDIA_CONTRAST": "orange",
                            "PERSONA": "violet",
                            "FACTUAL": "green",
                        }
                        color = intent_colors.get(intent, "gray")
                        st.markdown(f"**Detected Intent:** :{color}[{intent}]")

                        if analysis:
                            cols = st.columns(3)
                            with cols[0]:
                                st.markdown(f"**Politician:** `{analysis.get('politician', 'N/A')}`")
                            with cols[1]:
                                topics = ", ".join(analysis.get("topic_keywords", []))
                                st.markdown(f"**Topics:** `{topics or 'N/A'}`")
                            with cols[2]:
                                leaning = analysis.get("source_leaning") or "All"
                                st.markdown(f"**Media Filter:** `{leaning}`")

                    if not sources:
                        st.warning("No relevant statements found.")
                    else:
                        with st.expander(f"📄 View Retrieved Sources ({len(sources)})", expanded=False):
                            for res in sources:
                                meta = res.get("metadata", {})
                                score = res.get("score", 0.0)
                                doc_text = res.get("document", "")
                                title = meta.get("title", meta.get("year_month", ""))
                                source_name = meta.get("source", meta.get("sources", ""))
                                date = meta.get("publish_date", meta.get("year_month", ""))

                                if title:
                                    st.markdown(f"**{title}** (Score: {score:.4f})")
                                st.caption(f"Source: {source_name} | Date: {date}")
                                if doc_text:
                                    st.markdown(f"> {doc_text[:300]}{'...' if len(doc_text) > 300 else ''}")

                                url = meta.get("url")
                                if url:
                                    st.markdown(f"[Read Article]({url})")
                                st.divider()

                    st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
