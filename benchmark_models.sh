#!/bin/bash
# Benchmark script for embedding models
# Usage: ./benchmark_models.sh [--local] [--remote] [--output results.json] [--plot benchmark.png]

cd "$(dirname "$0")"

# Default parameters
LOCAL_ONLY=false
REMOTE_ONLY=false
OUTPUT_FILE=""
PLOT_FILE="benchmark_results.png"

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --local)
      LOCAL_ONLY=true
      shift
      ;;
    --remote)
      REMOTE_ONLY=true
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --plot)
      PLOT_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./benchmark_models.sh [--local] [--remote] [--output results.json] [--plot benchmark.png]"
      exit 1
      ;;
  esac
done

# Make sure the embedding service is running
if ! curl -s http://localhost:8000/health > /dev/null; then
  echo "Error: Embedding service is not running"
  echo "Please start it with: cd python/embedding_service && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
  exit 1
fi

# Get available models
MODELS=$(curl -s http://localhost:8000/models | jq -r '.models | keys[]')

if [ -z "$MODELS" ]; then
  echo "Error: No models found in the registry"
  exit 1
fi

# Filter models based on options
SELECTED_MODELS=""
for model in $MODELS; do
  if $LOCAL_ONLY; then
    # Only include local models (sentence_transformer type)
    TYPE=$(curl -s http://localhost:8000/models | jq -r ".models.\"$model\".type")
    if [ "$TYPE" == "sentence_transformer" ]; then
      SELECTED_MODELS="$SELECTED_MODELS $model"
    fi
  elif $REMOTE_ONLY; then
    # Only include remote models (not sentence_transformer type)
    TYPE=$(curl -s http://localhost:8000/models | jq -r ".models.\"$model\".type")
    if [ "$TYPE" != "sentence_transformer" ]; then
      SELECTED_MODELS="$SELECTED_MODELS $model"
    fi
  else
    # Include all models
    SELECTED_MODELS="$SELECTED_MODELS $model"
  fi
done

if [ -z "$SELECTED_MODELS" ]; then
  echo "Error: No models match the specified filter"
  exit 1
fi

echo "Benchmarking models: $SELECTED_MODELS"

# Construct the command
CMD="cd python/embedding_service && python -m benchmark --models$SELECTED_MODELS"

if [ ! -z "$OUTPUT_FILE" ]; then
  CMD="$CMD --output $OUTPUT_FILE"
fi

if [ ! -z "$PLOT_FILE" ]; then
  CMD="$CMD --plot $PLOT_FILE"
fi

# Run the benchmark
echo "Running benchmark: $CMD"
eval $CMD

echo "Benchmark completed!"
if [ ! -z "$PLOT_FILE" ]; then
  echo "Results plot saved to: $PLOT_FILE"
fi 