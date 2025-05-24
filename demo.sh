#!/bin/bash

# Vectron Demo Script
# This script demonstrates the core features of Vectron

# Set the API endpoint
API_URL="http://localhost:3000"
echo -e "\n\033[1;36m🚀 Vectron Vector Database Demo\033[0m"
echo -e "\033[1;36m==============================\033[0m"

# Check if server is running
echo -e "\n\033[1;33m1. Checking if Vectron server is running...\033[0m"
response=$(curl -s $API_URL)
if [ -z "$response" ]; then
  echo -e "\033[1;31m❌ Vectron server is not running. Please start it with 'cargo run --release'\033[0m"
  exit 1
else
  echo -e "\033[1;32m✅ Vectron server is running!\033[0m"
  echo -e "Response: $response"
fi

# Insert some vectors via text
echo -e "\n\033[1;33m2. Inserting vectors from text examples...\033[0m"

echo -e "\n\033[1;34m📝 Inserting vector for 'Machine Learning'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "doc1", "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}' \
  $API_URL/upsert-text | jq

echo -e "\n\033[1;34m📝 Inserting vector for 'Deep Learning'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "doc2", "text": "Deep learning is part of machine learning methods based on artificial neural networks with representation learning."}' \
  $API_URL/upsert-text | jq

echo -e "\n\033[1;34m📝 Inserting vector for 'Natural Language Processing'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "doc3", "text": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."}' \
  $API_URL/upsert-text | jq

echo -e "\n\033[1;34m📝 Inserting vector for 'Computer Vision'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "doc4", "text": "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos."}' \
  $API_URL/upsert-text | jq

echo -e "\n\033[1;34m📝 Inserting vector for 'Reinforcement Learning'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "doc5", "text": "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment to maximize cumulative reward."}' \
  $API_URL/upsert-text | jq

# List all vectors
echo -e "\n\033[1;33m3. Listing all stored vectors...\033[0m"
curl -s $API_URL/vectors | jq

# Demonstrate embedding generation
echo -e "\n\033[1;33m4. Generating an embedding from text...\033[0m"
echo -e "\033[1;34m🧠 Generating embedding for 'AI models can process text and images'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text": "AI models can process text and images"}' \
  $API_URL/embed | jq '.dimensions, .success, .message'

# Search by text
echo -e "\n\033[1;33m5. Searching for vectors similar to query text...\033[0m"

echo -e "\n\033[1;34m🔍 Searching for 'artificial intelligence applications'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text": "artificial intelligence applications"}' \
  $API_URL/search/text?top_k=3 | jq

echo -e "\n\033[1;34m🔍 Searching for 'computer understanding images'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text": "computer understanding images"}' \
  $API_URL/search/text?top_k=3 | jq

echo -e "\n\033[1;34m🔍 Searching for 'neural networks learning'\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"text": "neural networks learning"}' \
  $API_URL/search/text?top_k=3 | jq

# Run a small benchmark
echo -e "\n\033[1;33m6. Running a small benchmark...\033[0m"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"vector_count": 100, "dimension": 384, "search_count": 10}' \
  $API_URL/benchmark | jq

echo -e "\n\033[1;36m✨ Demo Complete! ✨\033[0m"
echo -e "Vectron has successfully demonstrated:"
echo -e "  - Vector insertion from text"
echo -e "  - Embedding generation"
echo -e "  - Similarity search"
echo -e "  - Benchmarking"
echo -e "\nFeel free to explore more capabilities using the API endpoints!" 