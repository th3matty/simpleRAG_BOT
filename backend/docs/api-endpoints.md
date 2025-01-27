# API Endpoints Documentation

This document provides detailed information about all available API endpoints in the SimpleRAG system.

## Chat Endpoints

### POST /chat

Process a chat request using RAG (Retrieval-Augmented Generation).

**Request Body:**

```json
{
  "query": "string"
}
```

**Response:**

```json
{
  "response": "string",
  "sources": [
    {
      "content": "string",
      "metadata": {
        "id": "string",
        "relevance_category": "string",
        "relevance_score": 0.0
      }
    }
  ],
  "metadata": {
    "model": "string",
    "finish_reason": "string",
    "usage": {}
  },
  "tool_used": "string",
  "tool_input": "string",
  "tool_result": "string"
}
```

## Document Management Endpoints

### GET /documents

Retrieve all documents from the database, organized by their original structure.

**Response:**

```json
{
  "count": 0,
  "documents": [
    {
      "document_id": "string",
      "title": "string",
      "source": "string",
      "chunks": [
        {
          "content": "string",
          "chunk_index": 0,
          "metadata": {},
          "embedding": []
        }
      ],
      "metadata": {
        "timestamp": "string",
        "tags": [],
        "total_chunks": 0,
        "file_type": "string"
      }
    }
  ]
}
```

### GET /documents/source/{source}

Retrieve all documents from a specific source.

**Parameters:**

- source (path): Source/filename to filter documents by (e.g., 'article1.md')

**Response:** Same as GET /documents

### POST /documents/upload/file

Upload and process a document file.

**Request Body (multipart/form-data):**

- file: File to upload
- title (optional): Document title
- source (optional, default: "file-upload"): Source identifier
- tags (optional): Document tags

**Response:**

```json
{
  "message": "string",
  "document_ids": ["string"],
  "metadata": {
    "timestamp": "string",
    "file_type": "string",
    "chunk_count": 0
  }
}
```

### PUT /documents/update/file

Update an existing document file or create if it doesn't exist.

**Request Body (multipart/form-data):**

- file: File to update
- title (optional): Document title
- source (optional, default: "file-upload"): Source identifier
- tags (optional): Document tags

**Response:**

```json
{
  "message": "string",
  "document_ids": ["string"],
  "metadata": {
    "timestamp": "string",
    "file_type": "string",
    "chunk_count": 0,
    "updated": true,
    "previous_version": {
      "chunk_count": 0,
      "timestamp": "string"
    },
    "changes": {
      "update_timestamp": "string"
    }
  }
}
```

### DELETE /documents/source/{source}

Delete all documents from a specific source.

**Parameters:**

- source (path): Source/filename to delete documents from (e.g., 'article1.md')

**Response:**

```json
{
  "message": "string",
  "deleted_count": 0,
  "source": "string",
  "metadata": {
    "timestamp": "string",
    "deleted_ids": ["string"]
  }
}
```

### DELETE /documents/collection

Delete all documents from the collection. This is a destructive operation and cannot be undone.

**Response:**

```json
{
  "message": "string",
  "deleted_count": 0,
  "source": "all",
  "metadata": {
    "timestamp": "string",
    "deleted_ids": ["string"]
  }
}
```

## Error Responses

All endpoints may return the following error responses:

### 500 Internal Server Error

```json
{
  "detail": "Error message describing what went wrong"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["string"],
      "msg": "string",
      "type": "string"
    }
  ]
}
```
