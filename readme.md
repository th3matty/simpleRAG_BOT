# RAG (Retrieval-Augmented Generation) System

A robust RAG system built with FastAPI, ChromaDB, and Anthropic's Claude for intelligent document retrieval and question answering.

## Features

- **Vector Database**: ChromaDB for efficient similarity search
- **Embeddings**: Sentence transformers for document and query embedding
- **LLM Integration**: Claude 3 for natural language generation
- **Tool System**: Extensible tool framework for enhanced capabilities
- **API Interface**: FastAPI with automatic documentation
- **Error Handling**: Comprehensive error handling and logging
- **Configuration**: Flexible configuration system with validation

### Implemented Tools

1. **Search Documents Tool**

   - Semantic search through document database
   - Automatic query embedding and similarity search
   - Returns relevant document snippets with metadata

2. **Calculator Tool**
   - Safe mathematical expression evaluation
   - Supports basic arithmetic operations
   - Error handling for invalid expressions

## Project Structure

```
backend/
├── app/
│   ├── routes/
│   │   └── chat.py         # API endpoints
│   ├── services/
│   │   ├── calculator.py   # Calculator tool implementation
│   │   ├── embeddings.py   # Embedding generation
│   │   ├── llm.py         # LLM interaction
│   │   └── tools.py       # Tool system and registry
│   ├── config.py          # Configuration management
│   ├── database.py        # Database operations
│   ├── exceptions.py      # Custom exceptions
│   └── main.py           # Application entry point
├── scripts/
│   └── load_test_data.py  # Data loading utility
├── tests/                # Test suite
├── .env.example         # Environment variables template
└── logs/               # Application logs
```

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd build_rag
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp backend/.env.example backend/.env
# Edit .env with your settings
```

Required environment variables:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- See `.env.example` for all available configuration options

## Usage

1. Start the server:

```bash
cd backend
uvicorn app.main:app --reload
```

2. The API will be available at `http://localhost:8000`

3. API Endpoints:

- POST `/chat`: Submit a query and get a response with sources
  - Automatically determines whether to use search or calculator
  - Returns tool usage details in response
- GET `/debug/documents`: List all documents in the database
- POST `/documents/upload`: Add documents to the database

### Example Queries

1. Information Retrieval:

```json
{
  "query": "What is the main product of our company?"
}
```

2. Mathematical Calculations:

```json
{
  "query": "What is 25 * 4 + 10?"
}
```

## API Documentation

Once the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

The system can be configured through environment variables:

### Model Settings

- `MODEL_NAME`: LLM model to use (default: claude-3-sonnet-20240229)
- `EMBEDDING_MODEL`: Embedding model (default: all-MiniLM-L6-v2)

### Chat Settings

- `MAX_CONTEXT_LENGTH`: Maximum context length (default: 2000)
- `TEMPERATURE`: LLM temperature (default: 0.7)
- `MAX_TOKENS`: Maximum response tokens (default: 500)
- `TOP_K_RESULTS`: Number of similar documents to retrieve (default: 3)

### Logging

- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_FILE`: Log file path (default: logs/app.log)

## Error Handling

The system implements a comprehensive error handling system:

- `RAGException`: Base exception for all custom exceptions
- `DatabaseError`: Database operation failures
- `EmbeddingError`: Embedding generation issues
- `LLMError`: LLM interaction problems
- `ConfigurationError`: Configuration validation errors
- `CalculatorError`: Mathematical expression evaluation errors

## Development

### Adding New Tools

The tool system is designed to be extensible. To add a new tool:

1. Create a new service in `app/services/`
2. Define the tool schema in `app/services/tools.py`
3. Implement the tool execution in `ToolExecutor`

### Adding New Documents

Use the load_test_data.py script:

```bash
python backend/scripts/load_test_data.py --input_dir path/to/documents
```

### Running Tests

```bash
pytest backend/tests/
```

## License

[MIT License](LICENSE)
