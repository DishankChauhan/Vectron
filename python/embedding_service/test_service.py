#!/usr/bin/env python3
"""
Test script for the Vectron multi-model embedding service.
This script sends requests to the service and verifies responses.
"""

import argparse
import json
import requests
import time
import sys
from typing import Dict, Any, List, Optional

# Default service URL
DEFAULT_URL = "http://localhost:8000"

# Test cases
TEST_TEXTS = [
    "This is a test of the embedding service.",
    "Vector databases are essential for semantic search.",
    "Embeddings capture the meaning of text in a numerical form."
]

def print_separator(title=None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print(f"\n{'-' * 3} {title} {'-' * (width - len(title) - 5)}")
    else:
        print(f"\n{'-' * width}")

def test_health(base_url: str) -> bool:
    """Test the health check endpoint."""
    print_separator("Health Check")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        print(f"Available models: {', '.join(data.get('available_models', []))}")
        return True
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_models(base_url: str) -> List[str]:
    """Test the models endpoint and return available model IDs."""
    print_separator("Available Models")
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        data = response.json()
        
        print(f"Default model: {data.get('default')}")
        models = data.get('models', {})
        
        if not models:
            print("No models available")
            return []
        
        print(f"Found {len(models)} models:")
        for model_id, config in models.items():
            model_type = config.get('type', 'unknown')
            model_name = config.get('model_name', 'unknown')
            dimensions = config.get('dimensions', 0)
            print(f"- {model_id}: {model_type} | {model_name} ({dimensions} dimensions)")
        
        return list(models.keys())
    except Exception as e:
        print(f"Models endpoint failed: {str(e)}")
        return []

def test_embed(base_url: str, model_id: Optional[str] = None) -> bool:
    """Test the embed endpoint with a single text."""
    model_suffix = f" with model {model_id}" if model_id else ""
    print_separator(f"Single Embedding{model_suffix}")
    
    test_text = TEST_TEXTS[0]
    try:
        # Prepare request URL and data
        url = f"{base_url}/embed"
        if model_id:
            url += f"?model_id={model_id}"
        
        payload = {"text": test_text}
        
        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        
        # Print results
        embedding = data.get('embedding', [])
        dimensions = data.get('dimensions', 0)
        model_info = data.get('model_info', {})
        
        print(f"Text: \"{test_text}\"")
        print(f"Embedding dimensions: {dimensions}")
        print(f"First 5 values: {embedding[:5]}...")
        print(f"Processing time: {data.get('processing_time_ms', 0):.2f} ms")
        print(f"Request time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Model info:")
        print(f"  - Name: {model_info.get('name')}")
        print(f"  - Provider: {model_info.get('provider')}")
        print(f"  - Remote: {model_info.get('is_remote', False)}")
        
        return True
    except Exception as e:
        print(f"Embed endpoint failed: {str(e)}")
        return False

def test_batch_embed(base_url: str, model_id: Optional[str] = None) -> bool:
    """Test the batch-embed endpoint with multiple texts."""
    model_suffix = f" with model {model_id}" if model_id else ""
    print_separator(f"Batch Embedding{model_suffix}")
    
    try:
        # Prepare request URL and data
        url = f"{base_url}/batch-embed"
        if model_id:
            url += f"?model_id={model_id}"
        
        payload = {"texts": TEST_TEXTS}
        
        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        
        # Print results
        embeddings = data.get('embeddings', [])
        count = data.get('count', 0)
        dimensions = data.get('dimensions', 0)
        model_info = data.get('model_info', {})
        
        print(f"Batch size: {count} texts")
        print(f"Embedding dimensions: {dimensions}")
        print(f"Processing time: {data.get('processing_time_ms', 0):.2f} ms")
        print(f"Request time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Model info:")
        print(f"  - Name: {model_info.get('name')}")
        print(f"  - Provider: {model_info.get('provider')}")
        print(f"  - Remote: {model_info.get('is_remote', False)}")
        
        # Calculate cosine similarity between first two embeddings as a sanity check
        if len(embeddings) >= 2:
            e1 = embeddings[0]
            e2 = embeddings[1]
            dot_product = sum(a * b for a, b in zip(e1, e2))
            norm_e1 = sum(a * a for a in e1) ** 0.5
            norm_e2 = sum(b * b for b in e2) ** 0.5
            similarity = dot_product / (norm_e1 * norm_e2)
            print(f"Similarity between first two texts: {similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"Batch-embed endpoint failed: {str(e)}")
        return False

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description='Test the Vectron embedding service')
    parser.add_argument('--url', type=str, default=DEFAULT_URL, help='Base URL of the service')
    parser.add_argument('--model', type=str, help='Specific model to test')
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')
    
    print(f"Testing embedding service at {base_url}")
    
    # Test health endpoint
    if not test_health(base_url):
        print("Health check failed, exiting")
        sys.exit(1)
    
    # Test models endpoint
    available_models = test_models(base_url)
    if not available_models:
        print("No models available, exiting")
        sys.exit(1)
    
    # Test embed endpoint with default model
    test_embed(base_url)
    
    # Test batch-embed endpoint with default model
    test_batch_embed(base_url)
    
    # Test with specific model if provided
    if args.model:
        if args.model in available_models:
            test_embed(base_url, args.model)
            test_batch_embed(base_url, args.model)
        else:
            print(f"Model {args.model} not available")
    
    print_separator()
    print("All tests completed successfully")

if __name__ == "__main__":
    main() 