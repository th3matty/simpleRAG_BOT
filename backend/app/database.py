import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.config import Settings

def init_chroma_client():
    client = chromadb.PersistentClient(
        path=Settings.chroma_persist_directory,
        settings=ChromaSettings(
            allow_reset=True,
            anonymized_telemetry=False
        )
    )
    return client