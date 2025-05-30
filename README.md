# 🚀 Vectron - Rust-Based Vector Database with Embedded AI

![Rust](https://img.shields.io/badge/Rust-🦀-orange)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

A high-performance, lightweight vector database built in Rust with text-to-embedding conversion, optimized vector storage, and similarity search capabilities. Vectron combines the speed of Rust with the power of modern AI embedding models.

## 🎬 Demo Video

[![Watch the demo]()](https://www.loom.com/share/e2880d78d93f4cc18944c1ee5674a874)


## ✨ Features

- **🧠 Multi-Model Embedding Support**: 
  - ✅ Local models (Sentence Transformers: MiniLM, BGE, etc.)
  - ✅ Remote APIs (OpenAI, Cohere, HuggingFace)
  - ✅ Model registry with configuration
  - ✅ Benchmarking and model comparison
- **🔀 Decoupled Architecture**:
  - ✅ Microservice for embedding generation
  - ✅ Standalone vector database service
  - ✅ Dockerized for easy deployment
- **⚡ High-Performance Vector Storage**: Fast in-memory storage with persistence capabilities
- **🔍 Similarity Search**: Efficient cosine similarity search with top-K results
- **🔄 Comprehensive REST API**: Full CRUD operations for vectors and embeddings
- **💾 Persistence Layer**: Save and load vectors to/from disk automatically
- **📊 Built-in Benchmarking**: Measure performance of vector operations and embedding models

## 🏗️ Architecture

```
                   ┌─────────────┐
                   │   Client    │
                   └──────┬──────┘
                          │
                          ▼
 ┌───────────────────────────────────────────┐
 │             Vectron REST API              │
 ├───────────┬─────────────────┬─────────────┤
 │ Embedding │  Vector Store   │ Benchmarks  │
 │ Client    │  CRUD & Search  │             │
 └─────┬─────┴────────┬────────┴─────────────┘
       │ HTTP         │
       ▼              │
┌──────────────┐  ┌───▼───────┐
│  Embedding   │  │Persistence│
│  Service     │  │  Layer    │
└──────┬───────┘  └───────────┘
       │
       ▼
┌──────────────┐
│ AI Models    │
│ Registry     │
└──────────────┘
```


## 📊 Model Benchmark Results

Example benchmark comparing embedding models:

| Model ID    | Provider           | Dimensions | Processing Time (s) | Contrast Score |
|-------------|-------------------|------------|---------------------|----------------|
| minilm      | sentence-transformers | 384     | 0.012              | 0.068          |
| bge-small   | sentence-transformers | 384     | 0.015              | 0.082          |
| openai-small| openai            | 1536       | 0.321              | 0.103          |
| cohere-english | cohere         | 384        | 0.456              | 0.089          |

## 🚀 Getting Started

### Running with Docker

The easiest way to get started is using Docker Compose:

```bash
docker-compose up
```

This will start both the Vectron database and the embedding service.

### Manual Setup

1. Clone the repository
2. Install Python dependencies:

```bash
cd python/embedding_service
pip install -r requirements.txt
```

3. Copy the environment template and add your API keys:

```bash
cp env.template .env
# Edit .env with your OpenAI, Cohere, and HuggingFace API keys
```

4. Start the embedding service:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

5. In a new terminal, build and run Vectron:

```bash
cargo run --release
```

The Vectron API will be available at `http://localhost:3000` and the embedding service at `http://localhost:8000`.

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
  -d '{"text": "sample document", "model_id": "openai-small"}' \
  -G -d 'top_k=3'
```

### List available embedding models

```bash
curl http://localhost:8000/models
```

### Benchmark embedding models

```bash
./benchmark_models.sh
```

## 🧪 Development Roadmap

This project has been developed in phases:

- ✅ Phase 0: Project Setup - Initial project structure
- ✅ Phase 1: Embedding Service - Decoupled embedding microservice
- ✅ Phase 2: Multi-Model Support - Multiple embedding providers and benchmarking
- 🔄 Phase 3: Advanced Vector Storage - Optimized in-memory storage with persistence
- 🔄 Phase 4: Search Algorithms - Similarity search optimization
- 🔄 Phase 5: Advanced Features - Clustering and segmentation
- 🔄 Phase 6: Performance Tuning - Testing and benchmarking

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
- OpenAI, Cohere, and HuggingFace for embedding APIs
- The Rust community for excellent libraries and tools
