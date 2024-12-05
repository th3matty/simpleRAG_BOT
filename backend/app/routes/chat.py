from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from ..services.llm import LLMService
from ..services.embeddings import EmbeddingService
from ..config import settings, logger

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Log incoming request
    logger.info(f"Received chat request: {request.query}")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService(settings.embedding_model)
        llm_service = LLMService(settings.anthropic_api_key)
        
        # Get query embedding
        query_embedding = embedding_service.get_embeddings([request.query])[0]
        logger.debug(f"Generated embedding of length: {len(query_embedding)}")
        
        # Search in Chroma
        chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collection = chroma_client.get_collection("documents")
        
        # Log collection info
        logger.info(f"Collection count: {collection.count()}")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
            
        # Generate response using context
        context = results['documents'][0]
        response = llm_service.generate_response(request.query, context)
        
        # Extract source strings from metadata
        sources = [meta['source'] for meta in results['metadatas'][0]]
        
        # Log the response
        logger.info(f"Generated response: {response[:100]}...")
        logger.info(f"Sources used: {sources}")
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add this utility endpoint to inspect the database
@router.get("/debug/documents")
async def get_documents():
    """Endpoint to inspect all documents in the database"""
    try:
        chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collection = chroma_client.get_collection("documents")
        
        # Get all documents
        results = collection.get()
        
        return {
            "count": collection.count(),
            "documents": [
                {
                    "content": doc,
                    "metadata": meta,
                }
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))