"""
Configuration module for the embedding service.
Loads environment variables and provides application settings.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    logger.info(f"Loading environment from {env_file}")
    load_dotenv(env_file)
else:
    logger.warning(f"No .env file found at {env_file}, using environment variables")

# Service configuration
PORT = int(os.environ.get("PORT", 8000))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# Configure logging level based on environment
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))

# Load available keys
available_keys = []
if OPENAI_API_KEY:
    available_keys.append("OpenAI")
if COHERE_API_KEY:
    available_keys.append("Cohere")
if HF_API_TOKEN:
    available_keys.append("HuggingFace")

if available_keys:
    logger.info(f"API keys available for: {', '.join(available_keys)}")
else:
    logger.warning("No API keys found in environment, only local models will work") 