from pydantic_settings import BaseSettings
import logging
import sys
from pathlib import Path

class Settings(BaseSettings):
    anthropic_api_key: str
    chroma_persist_directory: str = "./chroma_db"
    model_name: str = "claude-3-sonnet-20240229"
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    
    class Config:
        env_file = ".env"
        protected_namespaces = ()

settings = Settings()

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()