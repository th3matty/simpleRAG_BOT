from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat
from .core import config
from .services.document_processor import register_processors
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")


@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    logger.info("Registering document processors...")
    register_processors()
    logger.info("Document processors registered")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
