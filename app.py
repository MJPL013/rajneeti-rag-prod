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
    st.title("🗳️ Rajneeti RAG: Political News Analysis")

    # --- Initialize orchestrator ---
    try:
        orchestrator = get_orchestrator()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize RAG system: {e}")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("Configuration")

    # 1. Backend Engine Selection (driven by feature flags)
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
            "🔧 Backend Engine",
            range(len(engine_options)),
            format_func=lambda i: engine_options[i],
            help="Choose which database to query for retrieval.",
        )
        selected_engine = engine_keys[selected_idx]
    else:
        st.sidebar.info(f"🔧 Engine: {engine_options[0]}")
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

    # 4. LLM Info (read-only, from settings)
    st.sidebar.markdown(f"🤖 **LLM Provider:** `{orchestrator.llm_backend}`")
    st.sidebar.markdown(f"🌐 **Environment:** `{settings.APP_ENV}`")

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
            st.caption(f"🤖 {orchestrator.llm_backend.upper()} | 🔧 {selected_engine}")

            # Prepare filters
            filters = {}
            if filter_mode != "All":
                filters["classification"] = filter_mode

            with st.spinner("Retrieving & Analyzing..."):
                result = orchestrator.query(
                    query=prompt,
                    filters=filters,
                    engine=selected_engine,
                )

                sources = result["sources"]
                answer = result["answer"]

                if not sources:
                    st.warning("No relevant statements found.")
                else:
                    # Display Sources (Expandable)
                    with st.expander("View Retrieved Sources", expanded=False):
                        for res in sources:
                            meta = res["metadata"]
                            score = res["score"]
                            st.markdown(
                                f"**{meta.get('title', 'Unknown')}** (Score: {score:.4f})"
                            )
                            st.caption(
                                f"Source: {meta.get('source')} | Date: {meta.get('publish_date')}"
                            )
                            st.markdown(f"> {res['document']}")
                            st.markdown(
                                f"[Read Article]({meta.get('url')}) | ID: `{meta.get('article_id')}`"
                            )
                            st.divider()

                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    main()
