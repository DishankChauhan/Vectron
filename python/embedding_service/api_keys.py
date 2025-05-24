"""
API key management module for remote embedding providers.
Handles validation, rotation, and secure storage of API keys.
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Provider constants
PROVIDER_OPENAI = "openai"
PROVIDER_COHERE = "cohere"
PROVIDER_HUGGINGFACE = "huggingface"

# Storage settings
KEYS_FILE = Path(__file__).parent / "provider_keys.enc"
SALT = b'vectron_embedding_service_salt'  # Change in production!
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "vectron_default_encryption_key")  # Change in production!


def get_encryption_key() -> bytes:
    """Derive an encryption key from the environment variable."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(ENCRYPTION_KEY.encode()))


def save_keys(keys: Dict[str, Dict[str, str]]) -> None:
    """Save API keys to an encrypted file."""
    try:
        # Encrypt the keys
        cipher = Fernet(get_encryption_key())
        encrypted_data = cipher.encrypt(json.dumps(keys).encode())
        
        # Save to file
        with open(KEYS_FILE, "wb") as f:
            f.write(encrypted_data)
            
        logger.info(f"Saved encrypted API keys to {KEYS_FILE}")
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")


def load_keys() -> Dict[str, Dict[str, str]]:
    """Load API keys from an encrypted file."""
    # Default keys structure
    default_keys = {
        PROVIDER_OPENAI: {"api_key": os.environ.get("OPENAI_API_KEY", "")},
        PROVIDER_COHERE: {"api_key": os.environ.get("COHERE_API_KEY", "")},
        PROVIDER_HUGGINGFACE: {"api_key": os.environ.get("HF_API_TOKEN", "")}
    }
    
    if not KEYS_FILE.exists():
        # Create the keys file with default values from environment
        save_keys(default_keys)
        return default_keys
    
    try:
        # Read and decrypt the file
        with open(KEYS_FILE, "rb") as f:
            encrypted_data = f.read()
            
        cipher = Fernet(get_encryption_key())
        decrypted_data = cipher.decrypt(encrypted_data)
        
        return json.loads(decrypted_data)
    except Exception as e:
        logger.error(f"Error loading API keys: {e}, using environment variables")
        return default_keys


def get_api_key(provider: str) -> Optional[str]:
    """Get an API key for a specific provider."""
    keys = load_keys()
    provider_keys = keys.get(provider, {})
    return provider_keys.get("api_key")


def set_api_key(provider: str, api_key: str) -> None:
    """Set an API key for a specific provider."""
    keys = load_keys()
    
    if provider not in keys:
        keys[provider] = {}
        
    keys[provider]["api_key"] = api_key
    save_keys(keys)
    logger.info(f"Updated API key for {provider}")


def validate_api_key(provider: str, api_key: str) -> bool:
    """Validate an API key with a test request to the provider."""
    try:
        if provider == PROVIDER_OPENAI:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers
            )
            return response.status_code == 200
            
        elif provider == PROVIDER_COHERE:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.cohere.ai/v1/models",
                headers=headers
            )
            return response.status_code == 200
            
        elif provider == PROVIDER_HUGGINGFACE:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.get(
                "https://huggingface.co/api/models?limit=1",
                headers=headers
            )
            return response.status_code == 200
            
        else:
            logger.warning(f"Unknown provider: {provider}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating API key for {provider}: {e}")
        return False


def get_all_providers() -> List[Dict[str, Any]]:
    """Get information about all providers and their API key status."""
    keys = load_keys()
    providers = []
    
    for provider, provider_keys in keys.items():
        api_key = provider_keys.get("api_key", "")
        has_key = bool(api_key)
        
        providers.append({
            "id": provider,
            "name": provider.capitalize(),
            "has_api_key": has_key,
            "is_valid": validate_api_key(provider, api_key) if has_key else False
        })
        
    return providers


def get_provider_models(provider: str) -> List[Dict[str, Any]]:
    """Get available models from a provider."""
    api_key = get_api_key(provider)
    if not api_key:
        logger.warning(f"No API key available for {provider}")
        return []
        
    try:
        if provider == PROVIDER_OPENAI:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers
            )
            if response.status_code != 200:
                return []
                
            data = response.json()
            # Filter for embedding models
            return [
                {
                    "id": model["id"],
                    "name": model["id"],
                    "dimensions": 1536 if "3-small" in model["id"] else 3072  # Approximation
                }
                for model in data["data"]
                if "embedding" in model["id"]
            ]
            
        elif provider == PROVIDER_COHERE:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # Cohere doesn't have a direct models API, so we'll return known models
            return [
                {"id": "embed-english-v3.0", "name": "embed-english-v3.0", "dimensions": 1024},
                {"id": "embed-english-light-v3.0", "name": "embed-english-light-v3.0", "dimensions": 384},
                {"id": "embed-multilingual-v3.0", "name": "embed-multilingual-v3.0", "dimensions": 1024},
                {"id": "embed-multilingual-light-v3.0", "name": "embed-multilingual-light-v3.0", "dimensions": 384}
            ]
            
        elif provider == PROVIDER_HUGGINGFACE:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            # Return a selection of popular embedding models on HuggingFace
            return [
                {"id": "BAAI/bge-small-en-v1.5", "name": "BGE Small English", "dimensions": 384},
                {"id": "BAAI/bge-base-en-v1.5", "name": "BGE Base English", "dimensions": 768},
                {"id": "BAAI/bge-large-en-v1.5", "name": "BGE Large English", "dimensions": 1024},
                {"id": "intfloat/e5-small-v2", "name": "E5 Small", "dimensions": 384},
                {"id": "intfloat/e5-base-v2", "name": "E5 Base", "dimensions": 768},
                {"id": "intfloat/e5-large-v2", "name": "E5 Large", "dimensions": 1024}
            ]
            
        else:
            logger.warning(f"Unknown provider: {provider}")
            return []
            
    except Exception as e:
        logger.error(f"Error getting models for {provider}: {e}")
        return []


# Initialize API keys from environment variables on module load
if not KEYS_FILE.exists():
    default_keys = {
        PROVIDER_OPENAI: {"api_key": os.environ.get("OPENAI_API_KEY", "")},
        PROVIDER_COHERE: {"api_key": os.environ.get("COHERE_API_KEY", "")},
        PROVIDER_HUGGINGFACE: {"api_key": os.environ.get("HF_API_TOKEN", "")}
    }
    save_keys(default_keys) 