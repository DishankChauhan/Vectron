"""
Model registry and configuration for the embedding service.
Supports both local and remote embedding models.
"""

from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import time
import json
import os
from pathlib import Path
import logging
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "model_config.json"

class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    
    @abstractmethod
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """Generate an embedding for a single text."""
        pass
    
    @abstractmethod
    def batch_embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass
    
    @property
    def is_remote(self) -> bool:
        """Return whether this is a remote API model."""
        return False
    
    @property
    def provider(self) -> str:
        """Return the provider of this model."""
        return "unknown"
    
    def is_loaded(self) -> bool:
        """Return whether the model is loaded and ready."""
        return True
    
    @property
    def description(self) -> str:
        """Return a description of the model."""
        return f"{self.name} embedding model with {self.dimensions} dimensions"


class SentenceTransformerModel(EmbeddingModel):
    """Local sentence-transformer model implementation."""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None):
        """Initialize the model.
        
        Args:
            model_name: The name or path of the model
            model_path: Optional local path to the model
        """
        self._name = model_name
        self._model_path = model_path
        self._model = None
        self._dimensions = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load the model."""
        start_time = time.time()
        logger.info(f"Loading sentence transformer model: {self._name}")
        
        try:
            if self._model_path:
                self._model = SentenceTransformer(self._model_path)
            else:
                self._model = SentenceTransformer(self._name)
                
            # Get dimensions from a test embedding
            test_embedding = self._model.encode("test", convert_to_tensor=True)
            self._dimensions = len(test_embedding.tolist())
            
            load_time = time.time() - start_time
            logger.info(f"Model {self._name} loaded successfully in {load_time:.2f}s with {self._dimensions} dimensions")
        except Exception as e:
            logger.error(f"Failed to load model {self._name}: {str(e)}")
            raise
    
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """Generate an embedding for a single text."""
        if not self._model:
            self._load_model()
            
        embedding = self._model.encode(text, convert_to_tensor=True, normalize_embeddings=normalize)
        return embedding.tolist()
    
    def batch_embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._model:
            self._load_model()
            
        embeddings = self._model.encode(texts, convert_to_tensor=True, normalize_embeddings=normalize)
        return embeddings.tolist()
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimensions
    
    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self._name
    
    @property
    def provider(self) -> str:
        """Return the provider of this model."""
        return "sentence-transformers"


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI API-based embedding model implementation."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the model.
        
        Args:
            model_name: The name of the OpenAI embedding model
            api_key: OpenAI API key, defaults to OPENAI_API_KEY env var
        """
        self._name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._api_url = "https://api.openai.com/v1/embeddings"
        
        # Model dimensions
        self._dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        self._dimensions = self._dimensions_map.get(model_name, 1536)
        
        if not self._api_key:
            logger.warning("OpenAI API key not provided, embeddings will fail")
    
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """Generate an embedding for a single text."""
        if not self._api_key:
            raise ValueError("OpenAI API key not provided")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        
        payload = {
            "model": self._name,
            "input": text
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            
            # Normalize if requested (OpenAI embeddings are already normalized by default)
            return embedding
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def batch_embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._api_key:
            raise ValueError("OpenAI API key not provided")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        
        payload = {
            "model": self._name,
            "input": texts
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            # Normalize if requested (OpenAI embeddings are already normalized by default)
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimensions
    
    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self._name
    
    @property
    def is_remote(self) -> bool:
        """Return whether this is a remote API model."""
        return True
    
    @property
    def provider(self) -> str:
        """Return the provider of this model."""
        return "openai"


class CohereEmbeddingModel(EmbeddingModel):
    """Cohere API-based embedding model implementation."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the model.
        
        Args:
            model_name: The name of the Cohere embedding model
            api_key: Cohere API key, defaults to COHERE_API_KEY env var
        """
        self._name = model_name
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._api_url = "https://api.cohere.ai/v1/embed"
        
        # Model dimensions
        self._dimensions_map = {
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384
        }
        self._dimensions = self._dimensions_map.get(model_name, 1024)
        
        if not self._api_key:
            logger.warning("Cohere API key not provided, embeddings will fail")
    
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """Generate an embedding for a single text."""
        if not self._api_key:
            raise ValueError("Cohere API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self._name,
            "texts": [text],
            "input_type": "search_document"
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embeddings"][0]
            
            # Normalize if requested
            if normalize and embedding:
                embedding_np = np.array(embedding)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding = (embedding_np / norm).tolist()
            
            return embedding
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise
    
    def batch_embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._api_key:
            raise ValueError("Cohere API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self._name,
            "texts": texts,
            "input_type": "search_document"
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data["embeddings"]
            
            # Normalize if requested
            if normalize and embeddings:
                embeddings_np = np.array(embeddings)
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                non_zero_indices = norms > 0
                embeddings_np[non_zero_indices] = embeddings_np[non_zero_indices] / norms[non_zero_indices]
                embeddings = embeddings_np.tolist()
            
            return embeddings
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimensions
    
    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self._name
    
    @property
    def is_remote(self) -> bool:
        """Return whether this is a remote API model."""
        return True
    
    @property
    def provider(self) -> str:
        """Return the provider of this model."""
        return "cohere"


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """HuggingFace Inference API embedding model."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, dimensions: int = 768):
        """Initialize the model.
        
        Args:
            model_name: The name of the model on the HuggingFace Hub
            api_key: HuggingFace API token, defaults to HF_API_TOKEN env var
            dimensions: The output dimensions of the model
        """
        self._name = model_name
        self._api_key = api_key or os.environ.get("HF_API_TOKEN")
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._dimensions = dimensions
        
        if not self._api_key:
            logger.warning("HuggingFace API token not provided, embeddings will fail")
    
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """Generate an embedding for a single text."""
        if not self._api_key:
            raise ValueError("HuggingFace API token not provided")
        
        headers = {
            "Authorization": f"Bearer {self._api_key}"
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json={"inputs": text}
            )
            response.raise_for_status()
            embedding = response.json()
            
            # HF models return different formats, some return just the vector,
            # others return a list with a single vector, others return dict with features
            if isinstance(embedding, dict) and "features" in embedding:
                embedding = embedding["features"]
            elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = embedding[0]
            
            # Normalize if requested
            if normalize and embedding:
                embedding_np = np.array(embedding)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding = (embedding_np / norm).tolist()
            
            return embedding
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise
    
    def batch_embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._api_key:
            raise ValueError("HuggingFace API token not provided")
        
        headers = {
            "Authorization": f"Bearer {self._api_key}"
        }
        
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json={"inputs": texts}
            )
            response.raise_for_status()
            embeddings = response.json()
            
            # Handle different response formats
            if isinstance(embeddings, list) and all(isinstance(emb, list) for emb in embeddings):
                pass  # Already in the right format
            elif isinstance(embeddings, dict) and "features" in embeddings:
                embeddings = [embeddings["features"]]
            else:
                # For any other format, try to get embeddings one by one
                embeddings = [self.embed(text, normalize=False) for text in texts]
            
            # Normalize if requested
            if normalize and embeddings:
                embeddings_np = np.array(embeddings)
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                non_zero_indices = norms > 0
                embeddings_np[non_zero_indices] = embeddings_np[non_zero_indices] / norms[non_zero_indices]
                embeddings = embeddings_np.tolist()
            
            return embeddings
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._dimensions
    
    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self._name
    
    @property
    def is_remote(self) -> bool:
        """Return whether this is a remote API model."""
        return True
    
    @property
    def provider(self) -> str:
        """Return the provider of this model."""
        return "huggingface"


class ModelRegistry:
    """Registry of embedding models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            config_path: Path to the JSON config file
        """
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._models: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, EmbeddingModel] = {}
        self._default_model = ""
        
        self._load_config()
    
    def _load_config(self):
        """Load the model configuration from the JSON file."""
        # Default models config if file doesn't exist
        default_config = {
            "default_model": "minilm",
            "models": {
                "minilm": {
                    "type": "sentence_transformer",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384
                },
                "bge-small": {
                    "type": "sentence_transformer",
                    "model_name": "BAAI/bge-small-en",
                    "dimensions": 384
                }
            }
        }
        
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "r") as f:
                    config = json.load(f)
            else:
                logger.warning(f"Config file {self._config_path} not found, using default config")
                config = default_config
                
                # Save default config
                os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
                with open(self._config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
            
            self._models = config.get("models", {})
            self._default_model = config.get("default_model", next(iter(self._models.keys())))
            
            logger.info(f"Loaded {len(self._models)} models from config with default: {self._default_model}")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}, using default config")
            self._models = default_config["models"]
            self._default_model = default_config["default_model"]
    
    def get_model(self, model_id: Optional[str] = None) -> EmbeddingModel:
        """Get a model by ID, creating it if it doesn't exist.
        
        Args:
            model_id: The ID of the model to get, or None for the default
            
        Returns:
            An embedding model
        """
        model_id = model_id or self._default_model
        
        # Use cached instance if available
        if model_id in self._instances:
            return self._instances[model_id]
        
        # Check if model exists in config
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_config = self._models[model_id]
        model_type = model_config.get("type", "").lower()
        
        # Create model based on type
        if model_type == "sentence_transformer":
            model = SentenceTransformerModel(
                model_name=model_config["model_name"],
                model_path=model_config.get("model_path")
            )
        elif model_type == "openai":
            model = OpenAIEmbeddingModel(
                model_name=model_config["model_name"],
                api_key=model_config.get("api_key")
            )
        elif model_type == "cohere":
            model = CohereEmbeddingModel(
                model_name=model_config["model_name"],
                api_key=model_config.get("api_key")
            )
        elif model_type == "huggingface":
            model = HuggingFaceEmbeddingModel(
                model_name=model_config["model_name"],
                api_key=model_config.get("api_key"),
                dimensions=model_config.get("dimensions", 768)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cache the model instance
        self._instances[model_id] = model
        return model
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models.
        
        Returns:
            Dictionary of model configurations
        """
        # Filter out sensitive information like API keys
        safe_models = {}
        for model_id, config in self._models.items():
            safe_config = {**config}
            safe_config.pop("api_key", None)
            safe_models[model_id] = safe_config
        
        return {
            "default": self._default_model,
            "models": safe_models
        }
    
    def add_model(self, model_id: str, config: Dict[str, Any], save: bool = True) -> None:
        """Add a new model to the registry.
        
        Args:
            model_id: Unique identifier for the model
            config: Model configuration
            save: Whether to save the config to disk
        """
        if model_id in self._models:
            logger.warning(f"Overwriting existing model {model_id}")
        
        # Validate model config
        required_fields = ["type", "model_name"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in model config")
        
        self._models[model_id] = config
        
        # Clear cached instance if it exists
        if model_id in self._instances:
            del self._instances[model_id]
        
        if save:
            self._save_config()
    
    def remove_model(self, model_id: str, save: bool = True) -> None:
        """Remove a model from the registry.
        
        Args:
            model_id: The ID of the model to remove
            save: Whether to save the config to disk
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Don't allow removing the default model
        if model_id == self._default_model:
            raise ValueError(f"Cannot remove default model {model_id}")
        
        del self._models[model_id]
        
        # Clear cached instance if it exists
        if model_id in self._instances:
            del self._instances[model_id]
        
        if save:
            self._save_config()
    
    def set_default_model(self, model_id: str, save: bool = True) -> None:
        """Set the default model.
        
        Args:
            model_id: The ID of the model to set as default
            save: Whether to save the config to disk
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        self._default_model = model_id
        
        if save:
            self._save_config()
    
    def _save_config(self) -> None:
        """Save the current configuration to the JSON file."""
        config = {
            "default_model": self._default_model,
            "models": self._models
        }
        
        try:
            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved model config to {self._config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")


# Create a global model registry instance
registry = ModelRegistry()

# Public API functions that are imported by main.py
def load_models():
    """Load all models from config."""
    global registry
    registry = ModelRegistry()
    return registry

def get_model(model_id: Optional[str] = None) -> EmbeddingModel:
    """Get a model by ID or the default model."""
    return registry.get_model(model_id)

def list_models() -> List[Dict[str, Any]]:
    """List all available models."""
    return list(registry.list_models().values()) 