"""
Main FastAPI application for the Vectron embedding service.
Provides endpoints for generating and managing embeddings.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.security import OAuth2PasswordRequestForm, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from models import load_models, get_model, list_models, EmbeddingModel
import auth
import api_keys
import monitoring
import scaling
from vector_store import vector_store, IndexType, DistanceMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vectron Embedding Service",
    description="API for generating and managing vector embeddings",
    version="0.2.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Request and response models
class EmbeddingRequest(BaseModel):
    text: str
    normalize: bool = True


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    model_id: str
    text_hash: str
    processing_time: float


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model_id: str
    count: int
    processing_time: float


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_info: Dict[str, Any]


class ApiKeyResponse(BaseModel):
    api_key: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = auth.ROLE_USER


class CollectionCreate(BaseModel):
    name: str
    dimension: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.FLAT
    metadata: Optional[Dict[str, Any]] = None


class VectorItem(BaseModel):
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class VectorSearch(BaseModel):
    query_vector: List[float]
    top_k: int = 10
    filter: Optional[Dict[str, Any]] = None


# Authentication dependency
async def get_current_user_from_header(api_key: str = Depends(api_key_header), 
                               token_user: auth.User = Depends(auth.get_current_user)):
    """Get the current user from either API key or JWT token."""
    # First check JWT token
    if token_user:
        return token_user
    
    # Then check API key
    if api_key:
        api_user = auth.authenticate_api_key(api_key)
        if api_user:
            return api_user
    
    # If authentication is not enabled, return default user
    if not auth.AUTH_ENABLED:
        return auth.User(
            username="default",
            full_name="Default User",
            email="default@example.com",
            hashed_password="",
            role=auth.ROLE_ADMIN,
            disabled=False
        )
    
    return None


# Middleware to track request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request details
    path = request.url.path
    if path != "/health":  # Skip health check logging
        client_host = request.client.host if request.client else "unknown"
        status_code = response.status_code
        logger.info(f"{client_host} - {request.method} {path} - {status_code} in {process_time:.4f}s")
    
    return response


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting Vectron Embedding Service")
    
    # Initialize vector store
    try:
        # Make sure data directory exists
        os.makedirs("./data/vectors", exist_ok=True)
        
        # Check for existing collections
        collections = vector_store.list_collections()
        logger.info(f"Loaded {len(collections)} vector collections")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check API keys for providers
    provider_statuses = {}
    for provider in [api_keys.PROVIDER_OPENAI, api_keys.PROVIDER_COHERE, api_keys.PROVIDER_HUGGINGFACE]:
        key = api_keys.get_api_key(provider)
        provider_statuses[provider] = {
            "has_key": bool(key),
            "is_valid": api_keys.validate_api_key(provider, key) if key else False
        }
        monitoring.set_api_key_status(provider, provider_statuses[provider]["is_valid"])
    
    # Get model information
    models_info = []
    models_list = list_models()
    
    # Check if the result is a dictionary (from models.py)
    if isinstance(models_list, dict) and "models" in models_list:
        # Handle the case where list_models returns a dict with models key
        models_data = models_list["models"]
        for model_id, model_data in models_data.items():
            try:
                model_obj = get_model(model_id)
                models_info.append({
                    "id": model_id,
                    "name": model_data.get("model_name", model_id),
                    "loaded": hasattr(model_obj, "is_loaded") and model_obj.is_loaded() if model_obj else False,
                    "dimensions": model_data.get("dimensions", 0)
                })
            except Exception as e:
                logger.error(f"Error getting model info for {model_id}: {str(e)}")
    else:
        # Handle the case where list_models returns a list
        for model in models_list:
            if isinstance(model, dict) and "id" in model:
                try:
                    model_obj = get_model(model["id"])
                    models_info.append({
                        "id": model["id"],
                        "name": model.get("name", model["id"]),
                        "loaded": hasattr(model_obj, "is_loaded") and model_obj.is_loaded() if model_obj else False,
                        "dimensions": model.get("dimensions", 0)
                    })
                except Exception as e:
                    logger.error(f"Error getting model info for {model['id']}: {str(e)}")
    
    return {
        "status": "ok",
        "version": app.version,
        "providers": provider_statuses,
        "models": models_info,
        "auth_enabled": auth.AUTH_ENABLED,
        "vector_store": {
            "collections": len(vector_store.list_collections())
        }
    }


# Get embeddings for a single text
@app.post("/embed", response_model=EmbeddingResponse)
@monitoring.track_request_time(endpoint="embed")
async def get_embedding(
    request: EmbeddingRequest, 
    model_id: str = "minilm",
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Generate an embedding for a single text."""
    # Verify access to the model
    if current_user and not auth.can_use_model(model_id, current_user):
        raise HTTPException(status_code=403, detail=f"Not authorized to use model {model_id}")
    
    try:
        # Get the model
        model = get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Generate embedding with timing
        start_time = time.time()
        # Since the model functions are not async, use a threadpool to avoid blocking
        embedding = model.embed(request.text, normalize=request.normalize)
        process_time = time.time() - start_time
        
        # Track batch size and token count
        monitoring.track_batch_size(model_id, 1)
        
        # Calculate text hash for caching
        import hashlib
        text_hash = hashlib.md5(request.text.encode()).hexdigest()
        
        # Construct response
        return {
            "embedding": embedding if isinstance(embedding, list) else embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            "dimension": len(embedding),
            "model_id": model_id,
            "text_hash": text_hash,
            "processing_time": process_time
        }
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get embeddings for multiple texts
@app.post("/batch-embed", response_model=BatchEmbeddingResponse)
@monitoring.track_request_time(endpoint="batch-embed")
@scaling.cache_result(ttl=3600)
async def batch_embed(
    request: BatchEmbeddingRequest, 
    model_id: str = "minilm",
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Generate embeddings for multiple texts."""
    # Verify access to the model
    if current_user and not auth.can_use_model(model_id, current_user):
        raise HTTPException(status_code=403, detail=f"Not authorized to use model {model_id}")
    
    try:
        # Get the model
        model = get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Generate embeddings with timing
        start_time = time.time()
        # Since the model functions are not async, use a threadpool to avoid blocking
        embeddings = model.batch_embed(request.texts, normalize=request.normalize)
        process_time = time.time() - start_time
        
        # Track metrics
        monitoring.track_batch_size(model_id, len(request.texts))
        
        # Convert to list of lists if necessary
        if not isinstance(embeddings, list):
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
        else:
            embeddings_list = embeddings
        
        # Calculate embedding statistics for monitoring
        embedding_stats = monitoring.calculate_embedding_stats(embeddings_list)
        
        # Construct response
        return {
            "embeddings": embeddings_list,
            "dimension": embedding_stats.get("dimension", len(embeddings_list[0]) if embeddings_list else 0),
            "model_id": model_id,
            "count": len(embeddings_list),
            "processing_time": process_time
        }
    except Exception as e:
        logger.error(f"Batch embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# List available models
@app.get("/models")
async def get_models(current_user: auth.User = Depends(get_current_user_from_header)):
    """List all available models."""
    models = list_models()
    
    # If user is authenticated, filter models by access
    if auth.AUTH_ENABLED and current_user:
        role = auth.ROLES.get(current_user.role, auth.ROLES[auth.ROLE_USER])
        
        # Check if user has wildcard access
        if "*" in role.can_use_models:
            return models
        
        # Filter models
        return [model for model in models if model["id"] in role.can_use_models]
    
    return models


# Get details for a specific model
@app.get("/models/{model_id}")
async def get_model_details(
    model_id: str,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Get details for a specific model."""
    # Check model access
    if current_user and not auth.can_use_model(model_id, current_user):
        raise HTTPException(status_code=403, detail=f"Not authorized to use model {model_id}")
    
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "id": model_id,
        "name": model.name,
        "dimensions": model.dimension,
        "is_loaded": model.is_loaded(),
        "provider": model.provider,
        "description": model.description
    }


# Authentication endpoints
@app.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Generate a JWT token for authentication."""
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    token_data = {
        "sub": user.username,
        "role": user.role
    }
    access_token = auth.create_access_token(token_data)
    
    # User info for response
    user_info = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role
    }
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": user_info
    }


@app.post("/users", response_model=dict)
async def create_user(user: UserCreate, current_user: auth.User = Depends(auth.require_admin)):
    """Create a new user (admin only)."""
    try:
        new_user = auth.add_user(
            username=user.username,
            password=user.password,
            email=user.email,
            full_name=user.full_name,
            role=user.role
        )
        
        return {
            "username": new_user.username,
            "email": new_user.email,
            "full_name": new_user.full_name,
            "role": new_user.role,
            "status": "created"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/users/{username}/api-key", response_model=ApiKeyResponse)
async def generate_api_key_for_user(
    username: str, 
    current_user: auth.User = Depends(auth.require_admin)
):
    """Generate a new API key for a user (admin only)."""
    try:
        api_key = auth.generate_api_key(username)
        return {"api_key": api_key}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Provider API key management
@app.get("/providers")
async def list_providers(current_user: auth.User = Depends(auth.require_admin)):
    """List all embedding providers and their API key status."""
    return api_keys.get_all_providers()


@app.post("/providers/{provider}/api-key")
async def set_provider_api_key(
    provider: str, 
    api_key: str = Form(...),
    current_user: auth.User = Depends(auth.require_admin)
):
    """Set the API key for a provider."""
    # Validate the provider
    if provider not in [api_keys.PROVIDER_OPENAI, api_keys.PROVIDER_COHERE, api_keys.PROVIDER_HUGGINGFACE]:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    # Validate the API key
    is_valid = api_keys.validate_api_key(provider, api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid API key for {provider}")
    
    # Set the API key
    api_keys.set_api_key(provider, api_key)
    
    return {"status": "success", "provider": provider, "is_valid": True}


# Vector store endpoints
@app.post("/collections", response_model=dict)
async def create_collection(
    request: CollectionCreate,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Create a new vector collection."""
    try:
        config = vector_store.create_collection(
            name=request.name,
            dimension=request.dimension,
            distance_metric=request.distance_metric,
            index_type=request.index_type,
            metadata=request.metadata
        )
        
        return {
            "name": config.name,
            "dimension": config.dimension,
            "distance_metric": config.distance_metric,
            "index_type": config.index_type,
            "created_at": config.created_at,
            "status": "created"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections")
async def list_collections(current_user: auth.User = Depends(get_current_user_from_header)):
    """List all vector collections."""
    collections = vector_store.list_collections()
    return [
        {
            "name": c.name,
            "dimension": c.dimension,
            "distance_metric": c.distance_metric,
            "index_type": c.index_type,
            "created_at": c.created_at
        }
        for c in collections
    ]


@app.get("/collections/{collection_name}")
async def get_collection_info(
    collection_name: str,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Get information about a collection."""
    try:
        stats = vector_store.get_collection_stats(collection_name)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Delete a vector collection."""
    success = vector_store.delete_collection(collection_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
    
    return {"status": "deleted", "name": collection_name}


@app.post("/collections/{collection_name}/vectors")
async def add_vector(
    collection_name: str,
    vector_item: VectorItem,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Add a vector to a collection."""
    try:
        item_id = vector_store.add_item(
            collection_name=collection_name,
            vector=vector_item.vector,
            metadata=vector_item.metadata,
            item_id=vector_item.id
        )
        
        return {"id": item_id, "status": "created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{collection_name}/vectors/batch")
async def add_vectors_batch(
    collection_name: str,
    vector_items: List[VectorItem],
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Add multiple vectors to a collection."""
    try:
        vectors = [item.vector for item in vector_items]
        metadatas = [item.metadata for item in vector_items]
        ids = [item.id for item in vector_items]
        
        # Add items
        item_ids = vector_store.add_items(
            collection_name=collection_name,
            vectors=vectors,
            metadatas=metadatas,
            ids=ids
        )
        
        return {"ids": item_ids, "count": len(item_ids), "status": "created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{collection_name}/search")
@monitoring.track_request_time(endpoint="vector_search")
async def search_vectors(
    collection_name: str,
    search_params: VectorSearch,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Search for similar vectors in a collection."""
    try:
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=search_params.query_vector,
            top_k=search_params.top_k,
            filter=search_params.filter
        )
        
        return {
            "results": results,
            "count": len(results)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{collection_name}/vectors/{vector_id}")
async def get_vector(
    collection_name: str,
    vector_id: str,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Get a vector by ID."""
    try:
        item = vector_store.get_item(collection_name, vector_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Vector {vector_id} not found")
        
        return {
            "id": item.id,
            "vector": item.vector,
            "metadata": item.metadata,
            "created_at": item.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/collections/{collection_name}/vectors/{vector_id}")
async def delete_vector(
    collection_name: str,
    vector_id: str,
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Delete a vector by ID."""
    try:
        success = vector_store.delete_item(collection_name, vector_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Vector {vector_id} not found")
        
        return {"status": "deleted", "id": vector_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/embed-and-store")
async def embed_and_store(
    text: str = Form(...),
    model_id: str = Form("minilm"),
    collection_name: str = Form(...),
    metadata: Optional[str] = Form(None),
    current_user: auth.User = Depends(get_current_user_from_header)
):
    """Embed a text and store the vector in a collection."""
    # Check access
    if current_user and not auth.can_use_model(model_id, current_user):
        raise HTTPException(status_code=403, detail=f"Not authorized to use model {model_id}")
    
    try:
        # Parse metadata
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata")
        
        # Get the model
        model = get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Generate embedding
        embedding = model.embed(text, normalize=True)
        
        # Store in vector collection
        item_id = vector_store.add_item(
            collection_name=collection_name,
            vector=embedding if isinstance(embedding, list) else embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            metadata={
                "text": text,
                "model_id": model_id,
                "timestamp": time.time(),
                **metadata_dict
            }
        )
        
        return {
            "id": item_id,
            "collection": collection_name,
            "model_id": model_id,
            "status": "stored"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in embed_and_store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Run uvicorn server
if __name__ == "__main__":
    import uvicorn
    
    # Load models on startup
    load_models()
    
    # Start uvicorn server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        reload=os.environ.get("DEBUG", "false").lower() == "true"
    ) 