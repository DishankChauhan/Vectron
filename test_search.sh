#!/bin/bash

# Test script for Vectron Search API
# Make sure the server is running and test_embedding.sh has been run before
# to populate the database with test vectors

BASE_URL="http://localhost:3000"

echo "Testing Vectron Search API"
echo "=========================="

# First, add some test vectors if they don't exist yet
echo -e "\n1. Making sure we have test vectors"
echo -e "\nAdding vector for 'Machine learning models'"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"ml", "text":"Machine learning models can process vast amounts of data"}' \
  $BASE_URL/upsert-text > /dev/null

echo -e "\nAdding vector for 'Artificial intelligence'"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"ai", "text":"Artificial intelligence is transforming how we interact with technology"}' \
  $BASE_URL/upsert-text > /dev/null

echo -e "\nAdding vector for 'Vector databases'"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"vdb", "text":"Vector databases store embeddings for similarity search"}' \
  $BASE_URL/upsert-text > /dev/null

echo -e "\nAdding vector for 'Neural networks'"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"nn", "text":"Neural networks can learn complex patterns in data"}' \
  $BASE_URL/upsert-text > /dev/null

echo -e "\nAdding vector for 'Large language models'"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"llm", "text":"Large language models like GPT can generate human-like text"}' \
  $BASE_URL/upsert-text > /dev/null

echo -e "\nAdding vector for 'Data science'"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id":"ds", "text":"Data science involves extracting insights from structured and unstructured data"}' \
  $BASE_URL/upsert-text > /dev/null

# Wait a moment for all embeddings to be generated
sleep 2

# List all vectors
echo -e "\n2. Listing all available vectors"
curl -s $BASE_URL/vectors | jq

# Test search by text
echo -e "\n3. Searching by text (query: 'AI and machine learning')"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text":"AI and machine learning"}' \
  "$BASE_URL/search/text?top_k=3" | jq

# Get an embedding to use for vector search
echo -e "\n4. Getting embedding to use for vector search"
EMBEDDING=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text":"Neural networks and deep learning"}' \
  $BASE_URL/embed | jq -c '.embedding')

# Test search by vector
echo -e "\n5. Searching by vector (embedding for 'Neural networks and deep learning')"
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"vector\":$EMBEDDING}" \
  "$BASE_URL/search/vector?top_k=3" | jq

# Test with different query
echo -e "\n6. Searching with different query (query: 'databases for vector embeddings')"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text":"databases for vector embeddings"}' \
  "$BASE_URL/search/text?top_k=3" | jq

echo -e "\n\nSearch tests completed!" 