use std::time::{Duration, Instant};
use serde::Serialize;

/// Result of a benchmark operation
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Total time taken in milliseconds
    pub total_time_ms: f64,
    /// Average time per operation in milliseconds
    pub avg_time_ms: f64,
    /// Number of operations performed
    pub operations: usize,
}

/// Simple benchmark utility for measuring performance
pub struct Benchmark {
    /// Name of the benchmark
    name: String,
    /// Start time of the benchmark
    start_time: Option<Instant>,
    /// Total duration of all operations
    total_duration: Duration,
    /// Number of operations performed
    operations: usize,
}

impl Benchmark {
    /// Create a new benchmark with the given name
    pub fn new(name: &str) -> Self {
        Benchmark {
            name: name.to_string(),
            start_time: None,
            total_duration: Duration::from_secs(0),
            operations: 0,
        }
    }
    
    /// Start the benchmark timer
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }
    
    /// Stop the benchmark timer and record one operation
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            let duration = start.elapsed();
            self.total_duration += duration;
            self.operations += 1;
        }
    }
    
    /// Stop the benchmark timer and record multiple operations
    pub fn stop_multiple(&mut self, operations: usize) {
        if let Some(start) = self.start_time.take() {
            let duration = start.elapsed();
            self.total_duration += duration;
            self.operations += operations;
        }
    }
    
    /// Reset the benchmark
    pub fn reset(&mut self) {
        self.start_time = None;
        self.total_duration = Duration::from_secs(0);
        self.operations = 0;
    }
    
    /// Get the benchmark results
    pub fn results(&self) -> BenchmarkResult {
        let total_time_ms = self.total_duration.as_secs_f64() * 1000.0;
        let avg_time_ms = if self.operations > 0 {
            total_time_ms / self.operations as f64
        } else {
            0.0
        };
        
        BenchmarkResult {
            name: self.name.clone(),
            total_time_ms,
            avg_time_ms,
            operations: self.operations,
        }
    }
} 