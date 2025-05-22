# 🚀 Vectron - Rust-Based Vector Database with Embedded AI

![Rust](https://img.shields.io/badge/Rust-🦀-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

A high-performance, lightweight vector database built in Rust with text-to-embedding conversion, optimized vector storage, and similarity search capabilities. Vectron combines the speed of Rust with the power of modern AI embedding models.

## 🎬 Demo

## 📹 Demo Video

[![Watch the demo](https://cdn.loom.com/sessions/thumbnails/e2880d78d93f4cc18944c1ee5674a874-with-play.gif)](https://www.loom.com/share/e2880d78d93f4cc18944c1ee5674a874)


## ✨ Features

- **🧠 Text-to-Embedding Conversion**: Generate vector embeddings from text using HuggingFace models
- **⚡ High-Performance Vector Storage**: Fast in-memory storage with persistence capabilities
- **🔍 Similarity Search**: Efficient cosine similarity search with top-K results
- **🔄 Comprehensive REST API**: Full CRUD operations for vectors and embeddings
- **💾 Persistence Layer**: Save and load vectors to/from disk automatically
- **📊 Built-in Benchmarking**: Measure performance of vector operations

## 🏗️ Architecture

```
                  ┌─────────────┐
                  │   Client    │
                  └──────┬──────┘
                         │
                         ▼
┌───────────────────────────────────────────┐
│                 REST API                  │
├───────────┬─────────────────┬─────────────┤
│ Embedding │  Vector Store   │ Benchmarks  │
│ Generator │  CRUD & Search  │             │
└─────┬─────┴────────┬────────┴─────────────┘
      │              │
┌─────▼────┐   ┌────▼──────┐
│ Python   │   │ Persistence│
│ Models   │   │   Layer    │
└──────────┘   └────────────┘
```

## 📊 Benchmark Results

Benchmarks run on MacBook M1, 16GB RAM:

| Vectors | Dimension | Insert (ms/vector) | Search (ms/query) | Storage (MB) |
|---------|-----------|-------------------|------------------|--------------|
| 10      | 32        | 0.24              | 0.02             | <1           |
| 100     | 64        | 1.82              | 0.25             | <1           |
| 1,000   | 384       | 3.61              | 1.13             | 1.5          |
| 10,000  | 384       | 5.29              | 8.76             | 15           |

## 🚀 Getting Started

### Prerequisites

- Rust (latest stable version)
- Python 3.8+ (for embedding generation)

### Setup

1. Clone the repository
2. Install Python dependencies:

```bash
python -m pip install -r python/requirements.txt
```

### Running the Server

```bash
cargo run --release
```

The server will start on `http://localhost:3000`.

## 📚 API Usage Examples

### Insert a vector from text

```bash
curl -X POST http://localhost:3000/upsert-text \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "text": "This is a sample document for vector search"}'
```

### Search similar vectors by text

```bash
curl -X POST http://localhost:3000/search/text \
  -H "Content-Type: application/json" \
  -d '{"text": "sample document"}' \
  -G -d 'top_k=3'
```

### Get benchmark results

```bash
curl -X POST http://localhost:3000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"vector_count": 1000, "dimension": 384, "search_count": 50}'
```

## 🧪 Development Roadmap

This project has been developed in phases:

- ✅ Phase 0: Project Setup
- ✅ Phase 1: In-Memory Vector Store
- ✅ Phase 2: Embedding Generator
- ✅ Phase 3: Similarity Search
- ✅ Phase 4: REST API
- ✅ Phase 5: Testing and Benchmarking
- ✅ Phase 6: Persistence Layer

## 🔧 Future Enhancements

- Approximate search using HNSW or similar algorithms
- Clustering and visualization tools
- Authentication and access control
- Distributed vector storage for scaling
- Web UI for interactive demos

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- HuggingFace for their sentence-transformers models
- The Rust community for excellent libraries and tools 