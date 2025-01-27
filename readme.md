# RAG (Retrieval-Augmented Generation) System

A robust RAG system built with FastAPI, ChromaDB, and Anthropic's Claude for intelligent document retrieval and question answering.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Usage](#usage)
   - [Running the Server](#running-the-server)
   - [API Endpoints](#api-endpoints)
   - [Example Queries](#example-queries)
5. [Data Management](#data-management)
   - [Loading Test Data](#loading-test-data)
   - [Evaluating Chunking](#evaluating-chunking)
6. [API Documentation](#api-documentation)
7. [Configuration](#configuration)
   - [Model Settings](#model-settings)
   - [Chat Settings](#chat-settings)
   - [Logging](#logging)
8. [Error Handling](#error-handling)
9. [Development](#development)
   - [Adding New Tools](#adding-new-tools)
   - [Running Tests](#running-tests)
10. [License](#license)

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
│   │   └── chat.py                    # API endpoints
│   ├── services/
│   │   ├── calculator.py              # Calculator tool implementation
│   │   ├── embeddings.py              # Embedding generation
│   │   ├── llm.py                     # LLM interaction
│   │   ├── chunker.py                 # Text chunking service
│   │   ├── file_handler.py            # File operations
│   │   ├── document_processor/        # Document processors
│   │   │   ├── base.py               # Base processor class
│   │   │   ├── csv_processor.py      # CSV file processor
│   │   │   ├── docx_processor.py     # DOCX file processor
│   │   │   ├── factory.py            # Processor factory
│   │   │   └── pdf_processor.py      # PDF file processor
│   │   └── tools.py                   # Tool system and registry
│   ├── core/
│   │   ├── config.py                  # Configuration management
│   │   ├── database.py                # Database operations
│   │   ├── exceptions.py              # Custom exceptions
│   │   └── tools.py                   # Core tool utilities
│   └── main.py                        # Application entry point
├── data/
│   └── test_documents/                # Test document storage
│       ├── article1.md                # Sample markdown files
│       ├── article2.md
│       ├── article3.md
│       ├── article4.md
│       └── test_formats/              # Test files for processors
├── docs/                              # Documentation
│   ├── chunker.md
│   ├── client-query.md
│   └── document-upload.md
├── scripts/
│   ├── data_loading/                  # Data loading utilities
│   │   └── load_test_data.py
│   ├── evaluation/                    # Evaluation scripts
│   │   ├── evaluate_chunking.py
│   │   ├── evaluate_cluster_chunking.py
│   │   ├── evaluate_recursive_Langchain_char_chunking.py
│   │   └── evaluate_sliding_chunking.py
│   └── prompts/                       # Prompt templates
│       ├── output_for_test_cases.json
│       └── system_prompt_chatgpt_4o.md
├── tests/                             # Test suite
│   ├── conftest.py
│   ├── test_calculator.py
│   ├── test_csv_processor.py
│   ├── test_docx_processor.py
│   └── test_pdf_processor.py
├── .env.example                       # Environment template
└── logs/                             # Application logs
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

### Running the Server

1. Start the server:

```bash
cd backend
uvicorn app.main:app --reload
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

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

## Data Management

### Loading Test Data

The system includes a script to load test documents into the ChromaDB database. To use it:

1. Place your markdown documents in `backend/scripts/test_documents/`
2. Run the loading script:

```bash
cd backend
python scripts/data_loading/load_test_data.py
```

The script will:

- Create or recreate a "documents" collection in ChromaDB
- Load all markdown files from the test_documents directory
- Generate embeddings and store them in the database
- Display information about loaded documents

#### Testing Document Processors

The system includes processor tests for various document formats (PDF, CSV, and DOCX). These tests verify the processors' ability to extract content and metadata from different file types:

1. Place test files in `backend/data/test_documents/test_formats/`:

   - PDF files (\*.pdf)
   - CSV files (\*.csv)
   - DOCX files (\*.docx)

2. Run the processor tests:

```bash
cd backend
python -m pytest tests/test_pdf_processor.py
python -m pytest tests/test_csv_processor.py
python -m pytest tests/test_docx_processor.py
```

Each processor test validates:

- Text extraction from the document
- Metadata extraction capabilities
- Error handling for invalid files

Note: These tests only verify the processors' extraction capabilities. They do not load documents into the vector database. For loading documents into the database, use the data loading scripts in the `scripts/data_loading/` directory.

### Document Chunking Strategy

The system implements an adaptive semantic chunking strategy with the following features:

- **Content-Aware Chunking**: Analyzes document structure to identify:
  - Headers (Markdown and underlined)
  - Lists (ordered and unordered)
  - Code blocks (fenced and indented)
  - Paragraphs
- **Adaptive Sizing**:

  - Maximum chunk size: 1200 characters
  - Minimum chunk size: 400 characters
  - Overlap size: 200 characters
  - Dynamically adjusts chunk sizes based on content structure:
    - Smaller chunks for structured content (headers, lists)
    - Larger chunks for prose content
    - Preserves natural breaks in text

- **Semantic Preservation**:
  - Maintains context by keeping related content together
  - Respects document structure and natural boundaries
  - Ensures coherent chunks for better retrieval

### Chunking Evaluation System

The system includes comprehensive evaluation tools for assessing chunking performance:

#### Semantic Chunking Evaluation

Run the semantic chunking evaluation:

```bash
cd backend
python scripts/evaluation/evaluate_semantic_chunking.py
```

This script:

- Uses an external evaluation framework (chunking_evaluation)
- Compares our adaptive chunking with standardized metrics:
  - IOU (Intersection Over Union): Measures chunk boundary alignment
  - Recall: Evaluates information preservation
  - Standard deviation: Assesses consistency

#### Multiple Chunking Strategies

The system includes evaluation scripts for different chunking approaches:

- `evaluate_semantic_chunking.py`: Our adaptive semantic approach
- `evaluate_cluster_chunking.py`: Clustering-based chunking
- `evaluate_recursive_Langchain_char_chunking.py`: Langchain's recursive character chunking
- `evaluate_sliding_chunking.py`: Simple sliding window approach

Each script provides:

- Performance metrics for the specific strategy
- Comparative analysis with other methods
- Configuration options for tuning parameters

#### Evaluation Results

The evaluation framework tests chunking strategies against:

- Different document types and structures
- Various content lengths
- Multiple languages and formats

Results help optimize:

- Chunk size parameters
- Overlap settings
- Content-specific adjustments

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

### Running Tests

```bash
pytest backend/tests/
```

## License

[MIT License](LICENSE)
