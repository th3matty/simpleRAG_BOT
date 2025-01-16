from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat
from core import settings

# Create the FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
