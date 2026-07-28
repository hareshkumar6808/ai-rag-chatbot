"""
Production-Ready Streamlit AI PDF Chatbot Application.
Integrates Document Processor, ChromaDB Vector Store, and LangChain RAG Pipeline with Groq LLM.
"""

import os
import streamlit as st
from config.settings import settings
from src.utils.logger import get_logger
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGPipeline

logger = get_logger(__name__)

# Configure Streamlit Page
st.set_page_config(
    page_title="AI PDF Assistant | Production RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Display chat history

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # LangChain QA memory [(user, ai)]

    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager()

    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = None


def render_sidebar() -> None:
    """Render sidebar UI for PDF uploading, settings, and database management."""
    with st.sidebar:
        st.title("⚙️ System Control")
        st.markdown("---")

        # API Key & Model Status Check
        st.subheader("🔑 Model & API Status")
        provider = settings.LLM_PROVIDER.upper()
        st.info(f"**Provider**: {provider}\n\n**Model**: `{settings.MODEL_NAME}`")

        # API Key Override in Sidebar if missing in .env
        api_key_env = settings.GROQ_API_KEY if settings.LLM_PROVIDER == "groq" else settings.OPENAI_API_KEY
        is_key_set = bool(api_key_env and not api_key_env.startswith("your_") and not api_key_env.startswith("gsk_your_"))

        if not is_key_set:
            st.warning("⚠️ API Key not detected in `.env`")
            user_api_key = st.text_input(
                f"Enter {provider} API Key:",
                type="password",
                help=f"Enter your {provider} API Key to activate the LLM.",
            )
            if user_api_key:
                if settings.LLM_PROVIDER == "groq":
                    settings.GROQ_API_KEY = user_api_key.strip()
                else:
                    settings.OPENAI_API_KEY = user_api_key.strip()
                st.success("API Key applied for current session!")

        st.markdown("---")

        # Document Uploader Section
        st.subheader("📄 Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Select one or multiple PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload your PDF documents to chunk, embed, and query.",
        )

        # Advanced Chunking Sliders
        with st.expander("🛠️ Advanced Settings"):
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=300,
                max_value=2000,
                value=settings.CHUNK_SIZE,
                step=100,
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=500,
                value=settings.CHUNK_OVERLAP,
                step=50,
            )
            top_k = st.slider(
                "Retrieved Context Chunks (Top K)",
                min_value=1,
                max_value=10,
                value=settings.TOP_K_RESULTS,
                step=1,
            )

        # Process Documents Button
        if uploaded_files:
            if st.button("📥 Process & Embed PDFs", type="primary", use_container_width=True):
                with st.spinner("Processing PDF files & generating embeddings..."):
                    try:
                        processor = DocumentProcessor(
                            chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                        chunks, stats = processor.process_uploaded_files(uploaded_files)

                        added_count = st.session_state.vector_manager.add_documents(chunks)
                        st.session_state.doc_stats = stats

                        st.success(
                            f"✅ Processed {stats['total_files']} files ({stats['total_pages']} pages) "
                            f"into {added_count} vector embeddings!"
                        )
                    except Exception as e:
                        logger.error(f"Ingestion failed: {e}", exc_info=True)
                        st.error(f"Error processing files: {str(e)}")

        st.markdown("---")

        # Vector Database Status & Actions
        st.subheader("🗄️ Vector Store Status")
        doc_count = st.session_state.vector_manager.get_document_count()
        st.metric(label="Total Chunks in ChromaDB", value=doc_count)

        if doc_count > 0:
            if st.button("🗑️ Clear Vector Database", use_container_width=True):
                if st.session_state.vector_manager.clear_vector_store():
                    st.session_state.doc_stats = None
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.success("Vector store database cleared!")
                    st.rerun()

        st.markdown("---")
        st.caption("Powered by Streamlit, LangChain, ChromaDB & Groq")


def main() -> None:
    """Main Streamlit Application Entrypoint."""
    initialize_session_state()
    render_sidebar()

    # Main Header
    st.title("📚 AI PDF Chatbot (RAG)")
    st.markdown(
        "Upload your PDF documents in the sidebar and ask questions. "
        "The system retrieves relevant passages from **ChromaDB** and uses **Groq LLM** to provide precise answers with exact source page citations."
    )

    # Document Processing Summary Banner
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.info(
            f"📖 **Active Documents**: {', '.join(stats['file_names'])} | "
            f"**Pages**: {stats['total_pages']} | **Chunks**: {stats['total_chunks']}"
        )

    # Clear Chat Button
    if st.session_state.messages:
        col1, col2 = st.columns([0.85, 0.15])
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render sources accordion for assistant responses if available
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📌 Source Citations & References", expanded=False):
                    for idx, src in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**Source {idx}:** `{src['source']}` | **Page:** `{src['page']}`"
                        )
                        st.caption(f'"{src["content"]}"')
                        if idx < len(message["sources"]):
                            st.divider()

    # Chat Input Box
    if user_query := st.chat_input("Ask a question about your uploaded PDF documents..."):
        # 1. Check if vector store has documents
        total_vectors = st.session_state.vector_manager.get_document_count()
        if total_vectors == 0:
            st.warning("⚠️ Please upload and process at least one PDF document before asking questions.")
            return

        # 2. Check configuration validation
        valid, msg = settings.validate()
        if not valid:
            st.error(f"⚠️ Configuration Error: {msg}")
            return

        # 3. Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 4. Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    retriever = st.session_state.vector_manager.get_retriever(
                        top_k=settings.TOP_K_RESULTS
                    )
                    rag_pipeline = RAGPipeline(retriever=retriever)

                    result = rag_pipeline.answer_question(
                        question=user_query,
                        chat_history=st.session_state.chat_history,
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    # Display Answer
                    st.markdown(answer)

                    # Display Citations
                    if sources:
                        with st.expander("📌 Source Citations & References", expanded=False):
                            for idx, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**Source {idx}:** `{src['source']}` | **Page:** `{src['page']}`"
                                )
                                st.caption(f'"{src["content"]}"')
                                if idx < len(sources):
                                    st.divider()

                    # Save to Session State Memory
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                    st.session_state.chat_history.append((user_query, answer))

                except Exception as e:
                    logger.error(f"Error during RAG execution: {e}", exc_info=True)
                    st.error(f"An error occurred while answering your question: {str(e)}")


if __name__ == "__main__":
    main()
