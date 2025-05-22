# Vectron - Rust-Based Vector Database with Embedded AI

A performant, minimal vector database in Rust with text-to-embedding conversion, vector storage, and similarity search.

## Features

- Text-to-embedding conversion
- Vector storage
- Top-K similarity search
- REST interface

## Project Structure

```
src/
  ├── main.rs         # Application entry point
  ├── api/            # REST endpoints
  ├── db/             # In-memory vector store
  ├── embedding/      # Embedding logic
  └── utils/          # Similarity, helpers
```

## Getting Started

### Prerequisites

- Rust (latest stable version)
- Python 3.8+ (for embedding generation)

### Setup

1. Clone the repository
2. Configure variables in the `.env` file
3. Install Python dependencies:

```bash
python -m pip install -r python/requirements.txt
```

### Running the Server

Run the server:

```bash
cargo run
```

The server will start on `http://localhost:3000`.

## API Endpoints

- `GET /` - Health check endpoint
- `POST /vector` - Insert or update vector
- `GET /vector/:id` - Get vector by ID
- `DELETE /vector/:id` - Delete vector by ID
- `GET /vectors` - List all vector IDs
- `POST /embed` - Generate embedding from text
- `POST /upsert-text` - Create vector from text directly
- `POST /search/vector` - Search for similar vectors by vector
- `POST /search/text` - Search for similar vectors by text

## Development Roadmap

This project is being developed in phases:

- ✅ Phase 0: Project Setup 
- ✅ Phase 1: In-Memory Vector Store
- ✅ Phase 2: Embedding Generator
- ✅ Phase 3: Similarity Search
- ✅ Phase 4: REST API
- ⬜ Phase 5: Testing and Benchmarking
- ⬜ Phase 6: Persistence Layer (Optional) 