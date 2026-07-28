"""
Vector Store Manager Module.
Manages vector embeddings and persistent storage lifecycle using ChromaDB and HuggingFace Embeddings.
"""

import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Import HuggingFaceEmbeddings with backward-compatible fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

# Import Chroma with backward-compatible fallback
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages creation, ingestion, retrieval, and resetting of ChromaDB vector store."""

    COLLECTION_NAME = "pdf_chat_collection"

    def __init__(
        self,
        persist_directory: str = settings.CHROMA_PERSIST_DIR,
        embedding_model_name: str = settings.EMBEDDING_MODEL,
    ) -> None:
        """
        Initialize VectorStoreManager.

        Args:
            persist_directory (str): Local file path for ChromaDB storage.
            embedding_model_name (str): HuggingFace transformer model for generating embeddings.
        """
        self.persist_directory = os.path.abspath(persist_directory)
        self.embedding_model_name = embedding_model_name

        logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store: Optional[Chroma] = None

    def get_vector_store(self) -> Chroma:
        """
        Gets or initializes the Chroma vector store instance.

        Returns:
            Chroma: Persistent Chroma vector store.
        """
        if self._vector_store is None:
            logger.info(f"Loading persistent Chroma store at: {self.persist_directory}")
            os.makedirs(self.persist_directory, exist_ok=True)
            self._vector_store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> int:
        """
        Ingests document chunks into ChromaDB.

        Args:
            documents (List[Document]): Document chunks to store.

        Returns:
            int: Number of added documents.
        """
        if not documents:
            logger.warning("No documents provided for vector ingestion.")
            return 0

        vector_store = self.get_vector_store()
        logger.info(f"Ingesting {len(documents)} document chunks into ChromaDB...")
        vector_store.add_documents(documents)
        logger.info("Successfully ingested document chunks into ChromaDB.")
        return len(documents)

    def get_retriever(self, top_k: int = settings.TOP_K_RESULTS) -> VectorStoreRetriever:
        """
        Creates a LangChain retriever configured for MMR or Similarity search.

        Args:
            top_k (int): Number of top matching document chunks to retrieve.

        Returns:
            VectorStoreRetriever: LangChain retriever instance.
        """
        vector_store = self.get_vector_store()
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def get_document_count(self) -> int:
        """
        Gets total count of documents in the vector store collection.

        Returns:
            int: Document count.
        """
        try:
            vector_store = self.get_vector_store()
            collection = vector_store._collection
            return collection.count()
        except Exception as e:
            logger.warning(f"Could not retrieve document count: {str(e)}")
            return 0

    def clear_vector_store(self) -> bool:
        """
        Clears and resets the local ChromaDB database directory.

        Returns:
            bool: True if reset was successful.
        """
        try:
            self._vector_store = None
            if os.path.exists(self.persist_directory):
                logger.info(f"Removing Chroma database directory: {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
            logger.info("Vector store cleared successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}", exc_info=True)
            return False
