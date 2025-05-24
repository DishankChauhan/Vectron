from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
from sentence_transformers import SentenceTransformer
import torch
import os

# API models
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int
    text: str
    processing_time_ms: float

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int
    processing_time_ms: float

# Constants
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
AVAILABLE_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en",
}

# Initialize FastAPI app
app = FastAPI(
    title="Vectron Embedding Service",
    description="A microservice for generating text embeddings",
    version="1.0.0"
)

# Cache for models
model_cache: Dict[str, SentenceTransformer] = {}

def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load and cache the embedding model"""
    if model_name not in model_cache:
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Vectron Embedding Service API"}

@app.get("/models")
async def list_models():
    """List available embedding models"""
    return {
        "models": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL
    }

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(text_input: TextInput, model_name: Optional[str] = None):
    """Generate embedding for a single text input"""
    if not text_input.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    model_key = model_name if model_name in AVAILABLE_MODELS else DEFAULT_MODEL
    model = get_model(AVAILABLE_MODELS.get(model_key, DEFAULT_MODEL))
    
    start_time = time.time()
    embedding = model.encode(text_input.text, convert_to_tensor=True)
    embedding_list = embedding.tolist()
    end_time = time.time()
    
    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return EmbeddingResponse(
        embedding=embedding_list,
        dimensions=len(embedding_list),
        text=text_input.text,
        processing_time_ms=processing_time
    )

@app.post("/batch-embed", response_model=BatchEmbeddingResponse)
async def batch_embed_text(batch_input: BatchTextInput, model_name: Optional[str] = None):
    """Generate embeddings for multiple text inputs"""
    if not batch_input.texts:
        raise HTTPException(status_code=400, detail="Batch input cannot be empty")
    
    model_key = model_name if model_name in AVAILABLE_MODELS else DEFAULT_MODEL
    model = get_model(AVAILABLE_MODELS.get(model_key, DEFAULT_MODEL))
    
    start_time = time.time()
    embeddings = model.encode(batch_input.texts, convert_to_tensor=True)
    embeddings_list = embeddings.tolist()
    end_time = time.time()
    
    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return BatchEmbeddingResponse(
        embeddings=embeddings_list,
        dimensions=len(embeddings_list[0]) if embeddings_list else 0,
        count=len(embeddings_list),
        processing_time_ms=processing_time
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(model_cache.keys()),
        "memory_usage": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "N/A"
    }

if __name__ == "__main__":
    # Load the default model on startup
    get_model()
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the API server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) 