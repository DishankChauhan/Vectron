#!/usr/bin/env python3
"""
Text embedding generator using sentence-transformers.
This script takes text input and returns a vector embedding.
"""

import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore

# Default model - small but effective for most use cases
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions

def load_model(model_name=DEFAULT_MODEL):
    """Load the embedding model"""
    return SentenceTransformer(model_name)

def generate_embedding(text, model):
    """Generate embedding for the given text"""
    if not text.strip():
        return []
    
    # Generate embedding
    embedding = model.encode(text)
    return embedding.tolist()

def main():
    """Main entry point for the script"""
    # Check for input
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "No text provided. Usage: python embedding.py 'Your text here'",
            "embedding": []
        }))
        sys.exit(1)
    
    # Get input text
    text = sys.argv[1]
    
    try:
        # Load model
        model = load_model()
        
        # Generate embedding
        embedding = generate_embedding(text, model)
        
        # Return as JSON
        result = {
            "text": text,
            "dimensions": len(embedding),
            "embedding": embedding
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "embedding": []
        }))
        sys.exit(1)

if __name__ == "__main__":
    main() 