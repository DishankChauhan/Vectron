#!/bin/bash

# Test script for Vectron Persistence
# Make sure the server is running before executing this script

BASE_URL="http://localhost:3000"

echo "Testing Vectron Persistence"
echo "=========================="

# Insert some test vectors via text
echo -e "\n1. Inserting test vectors"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "persistent1", "text": "This is a test vector that should be saved to disk"}' \
  $BASE_URL/upsert-text | jq

curl -s -X POST -H "Content-Type: application/json" \
  -d '{"id": "persistent2", "text": "Another persistent vector that should survive restarts"}' \
  $BASE_URL/upsert-text | jq

# List vectors to verify insertion
echo -e "\n2. Listing vectors to verify insertion"
curl -s $BASE_URL/vectors | jq

echo -e "\n3. Now restart the server to test persistence"
echo "   After restarting, run:"
echo "   curl -s http://localhost:3000/vectors | jq"
echo
echo "   You should see the persistent1 and persistent2 vectors in the list"
echo "   This verifies that persistence is working correctly" 