use serde::{Deserialize, Serialize};
use axum::{extract::State, Json};
use tracing::info;
use std::sync::Arc;
use std::path::PathBuf;

use crate::api::AppStateInner;
use crate::utils::benchmark::{Benchmark, BenchmarkResult};
use crate::utils::random_vector;
use crate::db::VectorStore;

type AppState = Arc<AppStateInner>;

/// Benchmark parameters
#[derive(Debug, Deserialize)]
pub struct BenchmarkParams {
    /// Number of vectors to insert
    #[serde(default = "default_vector_count")]
    pub vector_count: usize,
    
    /// Dimension of the vectors
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    
    /// Number of search operations to perform
    #[serde(default = "default_search_count")]
    pub search_count: usize,
    
    /// Whether to test persistence
    #[serde(default)]
    pub test_persistence: bool,
}

fn default_vector_count() -> usize { 1000 }
fn default_dimension() -> usize { 384 }  // Default for our embedding model
fn default_search_count() -> usize { 100 }

/// Benchmark results
#[derive(Debug, Serialize)]
pub struct BenchmarkResponse {
    pub params: BenchmarkSummary,
    pub insert: BenchmarkResult,
    pub search: BenchmarkResult,
    pub persistence: Option<BenchmarkResult>,
}

/// Summary of benchmark parameters
#[derive(Debug, Serialize)]
pub struct BenchmarkSummary {
    pub vector_count: usize,
    pub dimension: usize,
    pub search_count: usize,
    pub test_persistence: bool,
}

/// Run benchmarks on the vector database
pub async fn run_benchmark(
    State(state): State<AppState>,
    Json(params): Json<BenchmarkParams>,
) -> Json<BenchmarkResponse> {
    info!("Running benchmark with {} vectors of dimension {}", 
          params.vector_count, params.dimension);
    
    // Prepare benchmarks
    let mut insert_bench = Benchmark::new("insert");
    let mut search_bench = Benchmark::new("search");
    let mut persistence_bench = Benchmark::new("persistence");
    
    // Generate random vectors for the benchmark
    let vectors: Vec<Vec<f32>> = (0..params.vector_count)
        .map(|_| random_vector(params.dimension))
        .collect();
    
    // Benchmark insertion
    {
        let mut store = state.store.write().unwrap();
        
        insert_bench.start();
        for (i, vector) in vectors.iter().enumerate() {
            store.insert(format!("bench_{}", i), vector.clone());
        }
        insert_bench.stop_multiple(params.vector_count);
    }
    
    // Benchmark search
    {
        let store = state.store.read().unwrap();
        
        for _ in 0..params.search_count {
            let query = random_vector(params.dimension);
            
            search_bench.start();
            let _ = store.search(&query, 10);
            search_bench.stop();
        }
    }
    
    // Benchmark persistence (optional)
    let persistence_result = if params.test_persistence {
        // Create a temporary vector store with persistence for testing
        let temp_dir = PathBuf::from("./data/benchmark_temp");
        let mut temp_store = match VectorStore::with_persistence(temp_dir.clone(), false) {
            Ok(store) => {
                info!("Created temporary store for persistence benchmark");
                store
            },
            Err(e) => {
                info!("Failed to create persistent store for benchmark: {}", e);
                return Json(BenchmarkResponse {
                    params: BenchmarkSummary {
                        vector_count: params.vector_count,
                        dimension: params.dimension,
                        search_count: params.search_count,
                        test_persistence: params.test_persistence,
                    },
                    insert: insert_bench.results(),
                    search: search_bench.results(),
                    persistence: None,
                });
            }
        };
        
        // Insert test data
        for (i, vector) in vectors.iter().take(params.vector_count).enumerate() {
            temp_store.insert(format!("temp_bench_{}", i), vector.clone());
        }
        
        // Benchmark save operation
        persistence_bench.start();
        let _ = temp_store.save();
        persistence_bench.stop();
        
        // Clean up
        std::fs::remove_dir_all(temp_dir).ok();
        
        Some(persistence_bench.results())
    } else {
        None
    };
    
    // Clean up benchmark vectors to avoid cluttering the database
    {
        let mut store = state.store.write().unwrap();
        for i in 0..params.vector_count {
            store.delete(&format!("bench_{}", i));
        }
    }
    
    // Return results
    Json(BenchmarkResponse {
        params: BenchmarkSummary {
            vector_count: params.vector_count,
            dimension: params.dimension,
            search_count: params.search_count,
            test_persistence: params.test_persistence,
        },
        insert: insert_bench.results(),
        search: search_bench.results(),
        persistence: persistence_result,
    })
} 