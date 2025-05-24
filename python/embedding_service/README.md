# Vectron Embedding Service

A high-performance embedding service for generating vector embeddings from text using multiple models. 
This service is part of the Vectron vector database system.

## Features

- **Multiple embedding models** supporting both local and remote providers:
  - Local models: Sentence Transformers (MiniLM, BGE, etc.)
  - Remote APIs: OpenAI, Cohere, Hugging Face
- **Authentication and authorization** with JWT tokens and API keys
- **Scaling capabilities** with caching, batching, and load balancing
- **Monitoring and metrics** with Prometheus and Grafana
- **Vector database** functionality with similarity search
- **Secure API key management** for remote providers

## Getting Started

### Prerequisites

- Docker and Docker Compose
- API keys for remote providers (OpenAI, Cohere, Hugging Face) if using remote models

### Quick Start

1. Clone the repository
2. Set up environment variables (or create a `.env` file)
3. Start the service:

```bash
docker-compose up -d
```

The service will be available at `http://localhost:8001`

## Configuration

### Environment Variables

Key environment variables:

```bash
# Authentication
AUTH_ENABLED=false  # Set to true to enable authentication
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# API Keys for remote providers
OPENAI_API_KEY=your-openai-key
COHERE_API_KEY=your-cohere-key
HF_API_TOKEN=your-huggingface-token

# Scaling & Performance
REDIS_ENABLED=true
REDIS_HOST=redis
MAX_WORKERS=4
MAX_BATCH_SIZE=64

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8002
```

### Model Configuration

Models are configured in `model_config.json`:

```json
{
    "default": "minilm",
    "models": {
        "minilm": {
            "type": "sentence_transformer",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "normalize": true
        },
        "bge-small": {
            "type": "sentence_transformer",
            "model_name": "BAAI/bge-small-en",
            "normalize": true
        },
        "openai-small": {
            "type": "openai",
            "model_name": "text-embedding-3-small"
        },
        "cohere-english": {
            "type": "cohere",
            "model_name": "embed-english-v3.0"
        },
        "huggingface": {
            "type": "huggingface",
            "model_name": "sentence-transformers/all-mpnet-base-v2"
        }
    }
}
```

## API Reference

### Authentication Endpoints

- `POST /token` - Get JWT token with username/password
- `POST /users` - Create new user (admin only)
- `POST /users/{username}/api-key` - Generate API key for user (admin only)

### Embedding Endpoints

- `POST /embed?model_id=minilm` - Generate embedding for single text
- `POST /batch-embed?model_id=minilm` - Generate embeddings for multiple texts
- `GET /models` - List available models
- `GET /models/{model_id}` - Get details for specific model

### Provider Management

- `GET /providers` - List providers and API key status
- `POST /providers/{provider}/api-key` - Set API key for provider

### Vector Database Endpoints

- `POST /collections` - Create a new vector collection
- `GET /collections` - List all collections
- `GET /collections/{collection_name}` - Get collection info
- `POST /collections/{collection_name}/vectors` - Add vector to collection
- `POST /collections/{collection_name}/search` - Search for similar vectors
- `POST /embed-and-store` - Embed text and store in collection

### Health and Monitoring

- `GET /health` - Service health check with detailed status

## Monitoring

The service includes a Prometheus metrics endpoint at `http://localhost:8002/metrics` and Grafana dashboards at `http://localhost:3000`.

Key metrics:
- Request counts and latencies
- Model loading times
- Memory and CPU usage
- Error rates
- Vector operation statistics

## Scaling

The service supports horizontal scaling through:
- Redis caching for embedding results
- Batch processing for efficient model usage
- Load balancing across model replicas
- Concurrent request handling

## Security

- JWT token authentication
- API key authorization
- Role-based access control
- Encrypted storage of provider API keys
- Request rate limiting by role

## Docker Compose Services

- `embedding-service`: Main embedding service
- `redis`: For caching embedding results
- `prometheus`: For metrics collection
- `grafana`: For metrics visualization

## Development

To run the service in development mode:

```bash
cd python/embedding_service
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Testing

Run the test client to verify functionality:

```bash
python test_client.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 