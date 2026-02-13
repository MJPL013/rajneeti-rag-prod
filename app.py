"""
Rajneeti RAG — Streamlit Frontend (Intent-Driven).

Supports:
  - Engine toggle: VectorRAG (ChromaDB) / GraphRAG (Neo4j)
  - Intent display when GraphRAG is active
  - Source cards with context from intent pipeline
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).parent))

from src.core.config import settings
from src.rag.orchestrator import RAGOrchestrator

st.set_page_config(page_title="Rajneeti RAG", layout="wide")


@st.cache_resource
def get_orchestrator():
    """Initialize the RAG orchestrator (cached across Streamlit reruns)."""
    return RAGOrchestrator()


def main():
    st.title("Rajneeti RAG: Political News Analysis")

    # --- Initialize orchestrator ---
    try:
        orchestrator = get_orchestrator()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("Configuration")

    # 1. Backend Engine Selection
    engine_options = []
    engine_keys = []
    if settings.ENABLE_VECTOR_RAG:
        engine_options.append("ChromaDB (Vector)")
        engine_keys.append("vector")
    if settings.ENABLE_GRAPH_RAG:
        engine_options.append("GraphRAG (Neo4j)")
        engine_keys.append("graph")

    if len(engine_options) > 1:
        selected_idx = st.sidebar.radio(
            "Backend Engine",
            range(len(engine_options)),
            format_func=lambda i: engine_options[i],
            help="Choose which database to query for retrieval.",
        )
        selected_engine = engine_keys[selected_idx]
    else:
        st.sidebar.info(f"Engine: {engine_options[0]}")
        selected_engine = engine_keys[0]

    # 2. Politician Selector
    raw_data_path = settings.RAW_DATA_DIR
    if raw_data_path.exists():
        politicians = [d.name for d in raw_data_path.iterdir() if d.is_dir()]
    else:
        politicians = []

    selected_politician = st.sidebar.selectbox(
        "Select Politician", ["All"] + politicians
    )

    # 3. Filter Mode
    filter_mode = st.sidebar.radio("Filter Mode", ["All", "Action", "Rhetoric"])

    st.sidebar.divider()

    # 4. Info
    st.sidebar.markdown(f"**LLM Provider:** `{orchestrator.llm_backend}`")
    st.sidebar.markdown(f"**Environment:** `{settings.APP_ENV}`")

    # ------------------------------------------------------------------
    # Chat Interface
    # ------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the politician..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant"):
            st.caption(f"LLM: {orchestrator.llm_backend.upper()} | Engine: {selected_engine}")

            # Prepare filters
            filters = {}
            if filter_mode != "All":
                filters["classification"] = filter_mode

            with st.spinner("Classifying intent & retrieving..."):
                result = orchestrator.query(
                    query=prompt,
                    filters=filters,
                    engine=selected_engine,
                )

                sources = result.get("sources", [])
                answer = result["answer"]
                intent = result.get("intent", None)
                analysis = result.get("analysis", None)

                # Intent badge (only for graph engine)
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
                            st.markdown(f"**Topics:** `{', '.join(analysis.get('topic_keywords', []))}`")
                        with cols[2]:
                            leaning = analysis.get('source_leaning') or 'All'
                            st.markdown(f"**Media Filter:** `{leaning}`")

                if not sources:
                    st.warning("No relevant statements found.")
                else:
                    with st.expander(f"View Retrieved Sources ({len(sources)})", expanded=False):
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
                            st.markdown(f"> {doc_text[:300]}...")

                            url = meta.get("url")
                            if url:
                                st.markdown(f"[Read Article]({url})")
                            st.divider()

                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    main()
