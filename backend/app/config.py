from pydantic_settings import BaseSettings
from pydantic import Field, validator
import logging
import sys
import os
from pathlib import Path
from typing import Optional
from .exceptions import ConfigurationError


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = Field(
        ..., description="Anthropic API key for Claude access"
    )

    # Database Settings
    chroma_persist_directory: str = Field(
        "./chroma_db", description="Directory for ChromaDB persistence"
    )
    PYTHONPATH: str

    # Model Settings
    model_name: str = Field("claude-3-sonnet-20240229", description="LLM model name")
    embedding_model: str = Field(
        "all-MiniLM-L6-v2", description="Sentence-transformers model name"
    )

    # Chat Settings
    max_context_length: int = Field(
        2000, description="Maximum length of context to send to LLM"
    )
    temperature: float = Field(
        0.7, description="Temperature for LLM responses", ge=0.0, le=1.0
    )
    max_tokens: int = Field(500, description="Maximum tokens in LLM response")
    top_k_results: int = Field(3, description="Number of similar documents to retrieve")

    # Logging Settings
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field("logs/app.log", description="Log file path")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level. Must be one of {valid_levels}"
            )
        return v.upper()

    class Config:
        env_file = os.getenv("ENV_FILE", ".env")
        protected_namespaces = ()


settings = Settings()


def setup_logging():
    """Configure logging with file and console handlers."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Get logging level from settings
        log_level = getattr(logging, settings.log_level)

        # Configure logging
        handlers = [logging.StreamHandler(sys.stdout)]

        if settings.log_file:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(exist_ok=True)
            handlers.append(logging.FileHandler(log_path))

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured with level: {settings.log_level}")
        return logger

    except Exception as e:
        raise ConfigurationError(f"Failed to setup logging: {str(e)}")


logger = setup_logging()
