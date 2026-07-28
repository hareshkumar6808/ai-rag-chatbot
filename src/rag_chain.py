"""
RAG Chain Module.
Orchestrates Retrieval-Augmented Generation using LangChain, Groq/OpenAI LLM,
history-aware retrieval, document context combination, and source citation extraction.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

# Robust imports for chains across different LangChain package versions
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    try:
        from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        from langchain_community.chains import create_history_aware_retriever, create_retrieval_chain  # type: ignore
        from langchain_community.chains.combine_documents import create_stuff_documents_chain  # type: ignore

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates LangChain RAG chain with context rephrasing, history, and source tracking."""

    def __init__(self, retriever: BaseRetriever) -> None:
        """
        Initialize RAG Pipeline with vector retriever.

        Args:
            retriever (BaseRetriever): Vector store retriever instance.
        """
        self.retriever = retriever
        self.llm = self._initialize_llm()
        self.rag_chain = self._build_rag_chain()

    def _initialize_llm(self) -> Any:
        """Initializes Chat model (ChatGroq or ChatOpenAI) based on configuration."""
        provider = settings.LLM_PROVIDER.lower()

        if provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in environment settings.")
            if ChatGroq is None:
                raise ImportError("langchain-groq is not installed.")

            logger.info(f"Initializing Groq LLM model: {settings.MODEL_NAME}")
            return ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE,
            )

        elif provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in environment settings.")
            if ChatOpenAI is None:
                raise ImportError("langchain-openai is not installed.")

            logger.info(f"Initializing OpenAI LLM model: {settings.MODEL_NAME}")
            return ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.MODEL_NAME,
                temperature=settings.TEMPERATURE,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _build_rag_chain(self) -> Any:
        """Constructs History-Aware Retriever and QA Retrieval Chain."""
        # 1. History-aware contextualizer prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 2. Question-Answering Prompt
        system_prompt = (
            "You are a helpful and precise AI assistant answering questions based on uploaded PDF documents.\n"
            "Use the provided context pieces below to answer the user's question.\n"
            "If the answer is not present in the context, explicitly state: 'I could not find relevant information in the uploaded PDF documents.'\n"
            "Keep your response accurate, structured, and clear.\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 3. Combine documents chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # 4. Final Retrieval Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        logger.info("RAG retrieval chain assembled successfully.")
        return rag_chain

    def answer_question(
        self, question: str, chat_history: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Executes RAG pipeline to generate answer and extract source citations.

        Args:
            question (str): User query text.
            chat_history (List[Tuple[str, str]]): Previous QA pairs [(user_msg, ai_msg), ...].

        Returns:
            Dict[str, Any]: Contains 'answer' (str) and 'sources' (List[dict]).
        """
        # Convert tuple chat history into LangChain BaseMessage objects
        formatted_history: List[BaseMessage] = []
        for human_msg, ai_msg in chat_history:
            formatted_history.append(HumanMessage(content=human_msg))
            formatted_history.append(AIMessage(content=ai_msg))

        logger.info(f"Processing question with {len(formatted_history)} history messages...")
        response = self.rag_chain.invoke(
            {
                "input": question,
                "chat_history": formatted_history,
            }
        )

        answer_text = response.get("answer", "No answer generated.")
        context_docs = response.get("context", [])

        # Format source citations
        sources: List[Dict[str, Any]] = []
        seen_citations = set()

        for doc in context_docs:
            source_name = doc.metadata.get("source_name", "Unknown File")
            page_num = doc.metadata.get("page", 1)
            content_snippet = doc.page_content.strip()

            citation_key = (source_name, page_num, content_snippet[:100])
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                sources.append(
                    {
                        "source": source_name,
                        "page": page_num,
                        "content": content_snippet,
                    }
                )

        return {
            "answer": answer_text,
            "sources": sources,
        }
