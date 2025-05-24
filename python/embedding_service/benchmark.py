#!/usr/bin/env python3
"""
Benchmark utility for embedding models in Vectron.
Compares different models for speed, dimensions, and embedding quality.
"""

import time
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from tabulate import tabulate
from models import registry, EmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample texts for benchmarking
SAMPLE_TEXTS = [
    "Artificial intelligence is reshaping industries across the global economy.",
    "Machine learning models can identify patterns invisible to humans.",
    "Vector databases store and retrieve high-dimensional data efficiently.",
    "Neural networks are designed to recognize patterns similar to the human brain.",
    "Large language models can generate human-like text based on prompts.",
    "Embeddings convert text, images, or audio into numerical vectors.",
    "Semantic search finds results based on meaning rather than keywords.",
    "Clustering algorithms group similar data points in multidimensional space.",
    "Recommendation systems use similarity metrics to suggest relevant items.",
    "Data preprocessing is crucial for machine learning model performance."
]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    return dot_product / (norm_v1 * norm_v2)

def run_similarity_test(model: EmbeddingModel) -> Dict[str, Any]:
    """Run similarity tests using the model"""
    # Generate embeddings for all sample texts
    embeddings = model.batch_embed(SAMPLE_TEXTS)
    
    # Calculate similarity matrix
    n = len(SAMPLE_TEXTS)
    similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    
    # Calculate avg similarity between related texts (domain-related)
    # Text indices 0-1-2-3-4 are related to AI/ML
    # Text indices 5-6-7-8-9 are related to embeddings/vectors
    related_ai_pairs = [(i, j) for i in range(5) for j in range(5) if i < j]
    related_vector_pairs = [(i, j) for i in range(5, 10) for j in range(5, 10) if i < j]
    
    ai_similarities = [similarity_matrix[i][j] for i, j in related_ai_pairs]
    vector_similarities = [similarity_matrix[i][j] for i, j in related_vector_pairs]
    
    # Calculate avg similarity between unrelated texts
    unrelated_pairs = [(i, j) for i in range(5) for j in range(5, 10)]
    unrelated_similarities = [similarity_matrix[i][j] for i, j in unrelated_pairs]
    
    return {
        "similarity_matrix": similarity_matrix,
        "ai_similarity_avg": sum(ai_similarities) / len(ai_similarities),
        "vector_similarity_avg": sum(vector_similarities) / len(vector_similarities),
        "unrelated_similarity_avg": sum(unrelated_similarities) / len(unrelated_similarities),
        "contrast_score": ((sum(ai_similarities) + sum(vector_similarities)) / 
                          (len(ai_similarities) + len(vector_similarities))) - 
                         (sum(unrelated_similarities) / len(unrelated_similarities))
    }

def benchmark_model(model_id: str) -> Dict[str, Any]:
    """Benchmark a single model"""
    logger.info(f"Benchmarking model: {model_id}")
    
    try:
        model = registry.get_model(model_id)
        
        # Single embedding speed test
        start_time = time.time()
        single_embedding = model.embed(SAMPLE_TEXTS[0])
        single_time = time.time() - start_time
        
        # Batch embedding speed test
        start_time = time.time()
        batch_embeddings = model.batch_embed(SAMPLE_TEXTS)
        batch_time = time.time() - start_time
        
        # Similarity tests
        similarity_results = run_similarity_test(model)
        
        return {
            "model_id": model_id,
            "name": model.name,
            "provider": model.provider,
            "is_remote": model.is_remote,
            "dimensions": model.dimensions,
            "single_embed_time": single_time,
            "batch_embed_time": batch_time,
            "batch_embed_avg_time": batch_time / len(SAMPLE_TEXTS),
            "contrast_score": similarity_results["contrast_score"],
            "ai_similarity_avg": similarity_results["ai_similarity_avg"],
            "vector_similarity_avg": similarity_results["vector_similarity_avg"],
            "unrelated_similarity_avg": similarity_results["unrelated_similarity_avg"]
        }
    except Exception as e:
        logger.error(f"Error benchmarking model {model_id}: {str(e)}")
        return {
            "model_id": model_id,
            "error": str(e)
        }

def run_benchmarks(model_ids: List[str] = None) -> List[Dict[str, Any]]:
    """Run benchmarks on all specified models or all available models"""
    # Get available models
    available_models = list(registry.list_models()["models"].keys())
    
    if not model_ids:
        model_ids = available_models
    else:
        # Validate model IDs
        for model_id in model_ids:
            if model_id not in available_models:
                logger.warning(f"Model {model_id} not found in registry, skipping")
                model_ids.remove(model_id)
    
    if not model_ids:
        logger.error("No valid models to benchmark")
        return []
    
    # Run benchmarks
    results = []
    for model_id in model_ids:
        try:
            result = benchmark_model(model_id)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark model {model_id}: {str(e)}")
    
    return results

def print_results(results: List[Dict[str, Any]]) -> None:
    """Print benchmark results in tabular format"""
    if not results:
        logger.info("No benchmark results to display")
        return
    
    # Filter any results with errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        logger.info("No valid benchmark results to display")
        return
    
    # Create table
    headers = [
        "Model ID", 
        "Provider", 
        "Dimensions", 
        "Single Time (s)", 
        "Batch Avg Time (s)", 
        "Contrast Score"
    ]
    
    table_data = []
    for r in valid_results:
        table_data.append([
            r["model_id"],
            r["provider"],
            r["dimensions"],
            f"{r['single_embed_time']:.4f}",
            f"{r['batch_embed_avg_time']:.4f}",
            f"{r['contrast_score']:.4f}"
        ])
    
    print("\nEmbedding Model Benchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print interpretation
    print("\nInterpretation:")
    print("- Single/Batch Time: Lower is better (faster)")
    print("- Contrast Score: Higher is better (better at differentiating related vs unrelated content)")
    print("- Dimensions: Trade-off between size and quality")

def plot_results(results: List[Dict[str, Any]], output_file: str = None) -> None:
    """Plot benchmark results"""
    if not results:
        logger.info("No benchmark results to plot")
        return
    
    # Filter any results with errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        logger.info("No valid benchmark results to plot")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time vs Dimensions
    models = [r["model_id"] for r in valid_results]
    dimensions = [r["dimensions"] for r in valid_results]
    batch_times = [r["batch_embed_avg_time"] for r in valid_results]
    is_remote = [r["is_remote"] for r in valid_results]
    
    # Use different colors for local vs remote models
    colors = ["blue" if not remote else "red" for remote in is_remote]
    
    ax1.scatter(dimensions, batch_times, c=colors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, model in enumerate(models):
        ax1.annotate(model, (dimensions[i], batch_times[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    ax1.set_title("Processing Time vs Dimensions")
    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Avg Batch Processing Time (s)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax1.scatter([], [], c="blue", label="Local")
    ax1.scatter([], [], c="red", label="Remote")
    ax1.legend()
    
    # Plot 2: Contrast Score vs Processing Time
    contrast_scores = [r["contrast_score"] for r in valid_results]
    
    ax2.scatter(batch_times, contrast_scores, c=colors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, model in enumerate(models):
        ax2.annotate(model, (batch_times[i], contrast_scores[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    ax2.set_title("Contrast Score vs Processing Time")
    ax2.set_xlabel("Avg Batch Processing Time (s)")
    ax2.set_ylabel("Contrast Score (higher = better)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_file}")
    else:
        plt.show()

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save benchmark results to a JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Benchmark embedding models')
    parser.add_argument('--models', type=str, nargs='+', help='Models to benchmark (default: all available)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--plot', type=str, help='Output file for benchmark plots')
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(args.models)
    
    # Print results
    print_results(results)
    
    # Save results if output file specified
    if args.output:
        save_results(results, args.output)
    
    # Plot results if plot file specified
    if results and (args.plot or not args.output):
        plot_results(results, args.plot)

if __name__ == "__main__":
    main() 