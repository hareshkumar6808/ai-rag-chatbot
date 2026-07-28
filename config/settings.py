"""
Configuration Settings Module.
Loads environment variables safely and exposes typed configuration settings for the RAG Chatbot.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """Centralized Application Settings."""

    def __init__(self) -> None:
        # API Keys
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

        # LLM Provider Configuration
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        self.MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip()
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

        # Embedding & Vector Database
        self.EMBEDDING_MODEL: str = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        self.CHROMA_PERSIST_DIR: str = os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma_db"
        ).strip()

        # Document Processing Parameters
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

        # Logging & Diagnostics
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper().strip()

    def validate(self) -> tuple[bool, str]:
        """Validate critical configuration settings before execution."""
        if self.LLM_PROVIDER == "groq":
            if not self.GROQ_API_KEY or self.GROQ_API_KEY == "gsk_your_groq_api_key_here" or self.GROQ_API_KEY == "your_groq_api_key_here":
                return False, "GROQ_API_KEY is not set or contains default placeholder. Please update your .env file."
        elif self.LLM_PROVIDER == "openai":
            if not self.OPENAI_API_KEY or "sk-" not in self.OPENAI_API_KEY:
                return False, "OPENAI_API_KEY is missing or invalid. Please set your OPENAI_API_KEY in .env file."
        return True, "Configuration is valid."


# Global settings singleton instance
settings = Settings()
