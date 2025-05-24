#!/usr/bin/env python3
"""
Test client for the multi-model embedding service.
Demonstrates how to interact with the service from Python.
"""

import argparse
import json
import requests
import time
import numpy as np
from typing import List, Dict, Any, Optional

# Default API endpoint
API_URL = "http://localhost:8001"

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Convert to numpy arrays for efficient computation
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def list_models(api_url: str = API_URL) -> Dict[str, Any]:
    """Get list of available models from the API."""
    response = requests.get(f"{api_url}/models")
    return response.json()

def embed_text(text: str, model_id: Optional[str] = None, api_url: str = API_URL) -> Dict[str, Any]:
    """Get embeddings for a single text string."""
    url = f"{api_url}/embed"
    if model_id:
        url += f"?model_id={model_id}"
    
    try:
        response = requests.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting embedding for model {model_id}: {e}")
        if hasattr(response, 'text'):
            print(f"Response: {response.text}")
        return {"error": str(e)}

def batch_embed(texts: List[str], model_id: Optional[str] = None, api_url: str = API_URL) -> Dict[str, Any]:
    """Get embeddings for multiple texts in one request."""
    url = f"{api_url}/batch-embed"
    if model_id:
        url += f"?model_id={model_id}"
    
    try:
        response = requests.post(url, json={"texts": texts})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting batch embeddings for model {model_id}: {e}")
        if hasattr(response, 'text'):
            print(f"Response: {response.text}")
        return {"error": str(e)}

def compare_models(text: str, model_ids: List[str], api_url: str = API_URL) -> None:
    """Compare embeddings across different models."""
    print(f"\nComparing models for text: '{text}'")
    print("-" * 80)
    
    embeddings = {}
    dimensions = {}
    valid_models = []
    
    # Get embeddings from each model
    for model_id in model_ids:
        try:
            print(f"Testing model: {model_id}...")
            start_time = time.time()
            result = embed_text(text, model_id, api_url)
            end_time = time.time()
            
            if "error" in result:
                print(f"  Error: {result['error']}")
                continue
                
            if "embedding" not in result:
                print(f"  Error: No embedding returned")
                continue
                
            embeddings[model_id] = result["embedding"]
            dimensions[model_id] = result["dimensions"]
            valid_models.append(model_id)
            
            print(f"Model: {model_id}")
            print(f"  Provider: {result['model_info']['provider']}")
            print(f"  Dimensions: {result['dimensions']}")
            print(f"  Service time: {result['processing_time_ms']:.2f}ms")
            print(f"  Total time: {(end_time - start_time) * 1000:.2f}ms")
            print(f"  First 5 values: {result['embedding'][:5]}")
            print()
        except Exception as e:
            print(f"  Failed to process model {model_id}: {e}")
    
    if len(valid_models) < 2:
        print("Not enough valid models to compare similarities.")
        return
        
    # Compare similarities across models
    print("Cross-model similarities:")
    for i, model_i in enumerate(valid_models):
        for j, model_j in enumerate(valid_models):
            if i < j:  # Only compute upper triangle of similarity matrix
                # We need to handle different dimensions by using the smaller dimension
                dim_i = dimensions[model_i]
                dim_j = dimensions[model_j]
                min_dim = min(dim_i, dim_j)
                
                # Use only the first min_dim dimensions for comparison
                vec_i = embeddings[model_i][:min_dim]
                vec_j = embeddings[model_j][:min_dim]
                
                try:
                    sim = cosine_similarity(vec_i, vec_j)
                    print(f"  {model_i} vs {model_j}: {sim:.4f} (using first {min_dim} dimensions)")
                except Exception as e:
                    print(f"  Error comparing {model_i} vs {model_j}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test client for the embedding service")
    parser.add_argument("--url", default=API_URL, help="API base URL")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--compare", action="store_true", help="Compare available models")
    parser.add_argument("--text", default="This is a test of the multi-model embedding system.", 
                        help="Text to embed")
    parser.add_argument("--local", action="store_true", help="Only test local models")
    
    args = parser.parse_args()
    api_url = args.url
    
    # Test API health
    try:
        health = requests.get(f"{api_url}/health")
        health.raise_for_status()
        health_data = health.json()
        
        print(f"Embedding Service v{health_data['version']} is {health_data['status']}")
        print(f"Available models: {', '.join(health_data['available_models'])}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return
    
    # Show model info
    models_data = list_models(api_url)
    print(f"\nDefault model: {models_data['default']}")
    print("\nAvailable Models:")
    for model_id, config in models_data["models"].items():
        print(f"- {model_id}: {config['model_name']} ({config['dimensions']}d, {config['type']})")
    
    if args.compare:
        # Get model list, filter if needed
        model_ids = list(models_data["models"].keys())
        if args.local:
            model_ids = [model_id for model_id, config in models_data["models"].items() 
                         if config["type"] == "sentence_transformer"]
            print(f"\nTesting only local models: {', '.join(model_ids)}")
        
        # Compare models
        compare_models(args.text, model_ids, api_url)
    else:
        # Test single embedding
        model_id = args.model
        print(f"\nTesting model: {model_id or 'default'}...")
        
        start_time = time.time()
        embedding_result = embed_text(args.text, model_id, api_url)
        end_time = time.time()
        
        if "error" in embedding_result:
            print(f"Error: {embedding_result['error']}")
            return
            
        print(f"Embedding with {model_id or 'default'} model:")
        print(f"Dimensions: {embedding_result['dimensions']}")
        print(f"Model: {embedding_result['model_info']['name']}")
        print(f"Processing time: {embedding_result['processing_time_ms']:.2f}ms")
        print(f"Total request time: {(end_time - start_time) * 1000:.2f}ms")
        print(f"First 5 values: {embedding_result['embedding'][:5]}")
        
        # Test batch embedding with same model
        texts = [
            args.text,
            "Vector databases use embeddings for semantic search.",
            "Different embedding models have different strengths."
        ]
        
        print("\nTesting batch embedding...")
        start_time = time.time()
        batch_result = batch_embed(texts, model_id, api_url)
        end_time = time.time()
        
        if "error" in batch_result:
            print(f"Error: {batch_result['error']}")
            return
            
        print(f"Generated {batch_result['count']} embeddings with {model_id or 'default'} model")
        print(f"Dimensions: {batch_result['dimensions']}")
        print(f"Processing time: {batch_result['processing_time_ms']:.2f}ms")
        print(f"Total request time: {(end_time - start_time) * 1000:.2f}ms")
        
        # Calculate similarities
        print("\nText similarities:")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                try:
                    sim = cosine_similarity(
                        batch_result["embeddings"][i],
                        batch_result["embeddings"][j]
                    )
                    print(f"  Text {i+1} vs Text {j+1}: {sim:.4f}")
                except Exception as e:
                    print(f"  Error calculating similarity: {e}")

if __name__ == "__main__":
    main() 