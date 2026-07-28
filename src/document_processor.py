"""
Document Processor Module.
Handles saving uploaded PDF files, extracting page content via LangChain PyPDFLoader,
and splitting text into chunks using RecursiveCharacterTextSplitter.
"""

import os
import tempfile
from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Processes PDF files into chunked LangChain Document objects with rich metadata."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        """
        Initialize DocumentProcessor with chunking parameters.

        Args:
            chunk_size (int): Target maximum characters per chunk.
            chunk_overlap (int): Number of overlapping characters between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_uploaded_files(
        self, uploaded_files: List[Any]
    ) -> Tuple[List[Document], dict]:
        """
        Processes a list of Streamlit uploaded PDF files.

        Args:
            uploaded_files (List[UploadedFile]): List of file-like objects from Streamlit uploader.

        Returns:
            Tuple[List[Document], dict]: Processed document chunks and ingestion statistics.
        """
        all_chunks: List[Document] = []
        total_pages = 0
        file_names = []

        temp_dir = tempfile.mkdtemp(prefix="rag_pdf_")
        logger.info(f"Created temporary directory for PDF processing: {temp_dir}")

        try:
            for uploaded_file in uploaded_files:
                original_filename = uploaded_file.name
                file_names.append(original_filename)

                # Save bytes to temporary file for PyPDFLoader
                temp_filepath = os.path.join(temp_dir, original_filename)
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                logger.info(f"Loading PDF: {original_filename}")
                loader = PyPDFLoader(temp_filepath)
                docs = loader.load()
                total_pages += len(docs)

                # Enhance metadata with clean filename and 1-indexed page numbers
                for doc in docs:
                    page_num = doc.metadata.get("page", 0) + 1
                    doc.metadata["source_name"] = original_filename
                    doc.metadata["page"] = page_num

                # Split document into chunks
                chunks = self.text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                logger.info(
                    f"Successfully processed '{original_filename}': "
                    f"{len(docs)} pages -> {len(chunks)} chunks."
                )

        except Exception as e:
            logger.error(f"Error processing PDF files: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to process PDF documents: {str(e)}") from e

        finally:
            # Clean up temporary directory
            try:
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup temp directory: {cleanup_err}")

        stats = {
            "total_files": len(uploaded_files),
            "file_names": file_names,
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
        }

        return all_chunks, stats
