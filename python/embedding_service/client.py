import requests
import sys
import json
from typing import List, Dict, Any, Optional

class EmbeddingServiceClient:
    """Client for interacting with the Vectron Embedding Service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def embed(self, text: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate embedding for a single text."""
        url = f"{self.base_url}/embed"
        if model:
            url += f"?model_name={model}"
            
        response = requests.post(
            url,
            json={"text": text}
        )
        
        if response.status_code != 200:
            error_message = f"Error from embedding service: {response.text}"
            return {"error": error_message, "embedding": []}
        
        return response.json()
    
    def batch_embed(self, texts: List[str], model: Optional[str] = None) -> Dict[str, Any]:
        """Generate embeddings for multiple texts."""
        url = f"{self.base_url}/batch-embed"
        if model:
            url += f"?model_name={model}"
            
        response = requests.post(
            url,
            json={"texts": texts}
        )
        
        if response.status_code != 200:
            error_message = f"Error from embedding service: {response.text}"
            return {"error": error_message, "embeddings": []}
        
        return response.json()

if __name__ == "__main__":
    """CLI interface for the client."""
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "No text provided. Usage: python client.py 'Your text here'",
            "embedding": []
        }))
        sys.exit(1)
    
    # Get input text and optional service URL
    text = sys.argv[1]
    service_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    try:
        client = EmbeddingServiceClient(service_url)
        result = client.embed(text)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "embedding": []
        }))
        sys.exit(1) 