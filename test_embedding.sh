#!/bin/bash

# Test script for Vectron Embedding API
# Make sure the server is running before executing this script
# Also ensure python dependencies are installed: pip install -r python/requirements.txt

BASE_URL="http://localhost:3000"

echo "Testing Vectron Embedding API"
echo "============================="

# Generate embedding for text
echo -e "\n1. Generate embedding for text"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text":"This is a test sentence for embedding generation"}' \
  $BASE_URL/embed | jq

# Directly create a vector from text
echo -e "\n2. Create vector from text"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"text1", "text":"This is another test sentence that will be stored as a vector"}' \
  $BASE_URL/upsert-text | jq

# Verify the vector was created
echo -e "\n3. Verify vector was created"
curl -s $BASE_URL/vector/text1 | jq

# Create more text vectors for search testing
echo -e "\n4. Create more text vectors"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"text2", "text":"Artificial intelligence is transforming how we interact with technology"}' \
  $BASE_URL/upsert-text | jq

curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"text3", "text":"Machine learning models can process vast amounts of data"}' \
  $BASE_URL/upsert-text | jq

curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"text4", "text":"Vector databases store embeddings for similarity search"}' \
  $BASE_URL/upsert-text | jq

# List all vectors
echo -e "\n5. List all vectors"
curl -s $BASE_URL/vectors | jq

echo -e "\n\nEmbedding tests completed!" 