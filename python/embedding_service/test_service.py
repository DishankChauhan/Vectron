#!/usr/bin/env python3
"""
Test script for the Vectron Embedding Service.
"""

import requests
import time
import json
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"  # Default URL for the embedding service

def test_single_embedding():
    """Test embedding generation for a single text."""
    print("\n1. Testing single text embedding...")
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/embed",
        json={"text": "This is a test sentence for embedding generation."}
    )
    
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success! Generated embedding with {data['dimensions']} dimensions")
        print(f"✓ Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"✓ API response time: {(end_time - start_time) * 1000:.2f}ms")
        
        # Print first 5 values of the embedding
        embedding_preview = [f"{x:.6f}" for x in data['embedding'][:5]]
        print(f"✓ Embedding preview: [{', '.join(embedding_preview)}...]")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")

def test_batch_embedding():
    """Test embedding generation for multiple texts."""
    print("\n2. Testing batch text embedding...")
    texts = [
        "This is the first test sentence.",
        "Here is another example sentence for embedding.",
        "Machine learning is transforming the world.",
        "Vector databases store embeddings for similarity search."
    ]
    
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/batch-embed",
        json={"texts": texts}
    )
    
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success! Generated {data['count']} embeddings with {data['dimensions']} dimensions each")
        print(f"✓ Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"✓ API response time: {(end_time - start_time) * 1000:.2f}ms")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")

def test_different_models():
    """Test embedding generation with different models."""
    print("\n3. Testing different embedding models...")
    
    # First, get available models
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        models = response.json()
        print(f"✓ Available models: {list(models['models'].keys())}")
        
        # Test each model
        for model_name in models['models'].keys():
            print(f"\n  Testing model: {model_name}")
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/embed?model_name={model_name}",
                json={"text": "This is a test for different embedding models."}
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Success! Generated embedding with {data['dimensions']} dimensions")
                print(f"  ✓ Processing time: {data['processing_time_ms']:.2f}ms")
                print(f"  ✓ API response time: {(end_time - start_time) * 1000:.2f}ms")
            else:
                print(f"  ✗ Error: {response.status_code} - {response.text}")
    else:
        print(f"✗ Error getting models: {response.status_code} - {response.text}")

def test_health_check():
    """Test the health check endpoint."""
    print("\n4. Testing health check...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Service status: {data['status']}")
        print(f"✓ Models loaded: {data['models_loaded']}")
        print(f"✓ Memory usage: {data['memory_usage']}")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("Testing Vectron Embedding Service")
    print("================================")
    
    try:
        # First, check if the service is running
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print(f"✓ Service is running: {response.json()['message']}")
            
            # Run all tests
            test_single_embedding()
            test_batch_embedding()
            test_different_models()
            test_health_check()
            
            print("\nAll tests completed!")
        else:
            print(f"✗ Service is not responding properly: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to the embedding service at {BASE_URL}")
        print("  Make sure the service is running before executing this script.") 