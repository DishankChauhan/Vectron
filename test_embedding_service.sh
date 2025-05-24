#!/bin/bash

# Test script for Vectron using the Embedding Service
# Make sure the embedding service is running on port 8000 before executing this script
# Also ensure the main Vectron service is running on port 3000

EMBEDDING_URL="http://localhost:8000"
VECTRON_URL="http://localhost:3000"

echo "Testing Vectron with Embedding Service"
echo "======================================"

# First, check if the embedding service is running
if ! curl -s "$EMBEDDING_URL" > /dev/null; then
  echo "❌ Embedding service is not running at $EMBEDDING_URL"
  exit 1
fi

echo "✅ Embedding service is running at $EMBEDDING_URL"

# Check models available in the embedding service
echo -e "\n1. Checking available embedding models:"
curl -s "$EMBEDDING_URL/models" | jq

# Generate embedding directly from the embedding service
echo -e "\n2. Generate embedding directly from embedding service:"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text":"This is a test sentence for embedding generation"}' \
  "$EMBEDDING_URL/embed" | jq '.dimensions, .processing_time_ms'

# Create a vector from text using the main Vectron API
echo -e "\n3. Create vector from text using Vectron API (which should use the embedding service):"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"test1", "text":"This is a test sentence that should use the embedding service"}' \
  "$VECTRON_URL/upsert-text" | jq

# Verify the vector was created
echo -e "\n4. Verify the vector was created:"
curl -s "$VECTRON_URL/vector/test1" | jq

# Now search for similar vectors
echo -e "\n5. Search for similar vectors:"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text":"This is a similarity search test"}' \
  "$VECTRON_URL/search/text" | jq

echo -e "\nTests completed!" 