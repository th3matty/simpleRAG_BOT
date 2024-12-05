# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, ChromaDB, and Claude AI. This chatbot can answer questions based on your documents by combining document retrieval with AI-powered text generation.

## Project Structure

```
backend/
├── app/
│   ├── init.py
│   ├── main.py           # FastAPI app
│   ├── config.py         # Configuration
│   ├── database.py       # Chroma setup
│   ├── routes/
│   │   └── chat.py       # Chat endpoints
│   └── services/
│       ├── llm.py        # LLM integration
│       └── embeddings.py  # Text processing
├── scripts/
│   └── load_test_data.py # Test data loader
├── tests/
├── .env
└── requirements.txt
```

## Prerequisites

- Python 3.10 or higher
- Anthropic API key (for Claude)

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd rag-chatbot
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac

or
.venv\Scripts\activate    # On Windows
```

Install dependencies:

```bash
pip install fastapi uvicorn anthropic chromadb pydantic-settings python-dotenv sentence-transformers
```

Create a .env file in the backend directory:

```bash
cd backend
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
echo "CHROMA_PERSIST_DIRECTORY=./chroma_db" >> .env
```

Replace `your-api-key-here` with your actual Anthropic API key.

## Running the Application

Start the FastAPI server:

```bash
cd backend
uvicorn app.main:app --reload
```

Load test documents (optional):

```bash
python scripts/load_test_data.py
```

The server will be running at `http://localhost:8000\`

## Usage

### API Endpoints

#### Chat Endpoint

- URL: `/api/chat`
- Method: `POST`
- Request Body:

```json
{
  "query": "Your question here"
}
```

### Testing the API

Using Swagger UI:

- Open `http://localhost:8000/docs\` in your browser
- Try out the chat endpoint
  Using curl:

```bash
curl -X POST "http://localhost:8000/api/chat"
-H "Content-Type: application/json"
-d '{"query": "What is Python?"}'
```

Using Python requests:

```python
import requests

response = requests.post(
"http://localhost:8000/api/chat",
json={"query": "What is Python?"}
)
print(response.json())
```

## Features

- Document retrieval using ChromaDB
- AI-powered responses using Claude
- Sentence transformer embeddings
- FastAPI backend with async support
- CORS middleware for frontend integration
- Environment-based configuration

## Contributing

Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Create a Pull Request

## License

[Your chosen license]

## TODO

- [ ] Add frontend implementation
- [ ] Add more document processors
- [ ] Implement caching
- [ ] Add authentication
- [ ] Add rate limiting
- [ ] Add more test coverage
- [ ] Add Notion integration
