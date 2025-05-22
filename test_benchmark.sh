#!/bin/bash

# Test script for Vectron Benchmarking
# Make sure the server is running before executing this script

BASE_URL="http://localhost:3000"

echo "Running Vectron Benchmarks"
echo "=========================="

# Small benchmark (1,000 vectors)
echo -e "\n1. Running small benchmark (1,000 vectors)"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"vector_count": 1000, "dimension": 384, "search_count": 100}' \
  $BASE_URL/benchmark | jq

# Medium benchmark (10,000 vectors)
echo -e "\n2. Running medium benchmark (10,000 vectors)"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"vector_count": 10000, "dimension": 384, "search_count": 100}' \
  $BASE_URL/benchmark | jq

# Persistence benchmark (5,000 vectors)
echo -e "\n3. Running persistence benchmark (5,000 vectors)"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"vector_count": 5000, "dimension": 384, "search_count": 50, "test_persistence": true}' \
  $BASE_URL/benchmark | jq

# Large benchmark (50,000 vectors) - only if you have enough RAM
echo -e "\n4. Running large benchmark (50,000 vectors) - this may take a while"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"vector_count": 50000, "dimension": 384, "search_count": 100}' \
  $BASE_URL/benchmark | jq

echo -e "\n\nBenchmark tests completed!" 