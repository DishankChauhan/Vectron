use std::process::Command;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{info, warn, error};

/// Response format from Python embedding script
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub text: Option<String>,
    pub dimensions: Option<usize>,
    pub embedding: Vec<f32>,
    pub error: Option<String>,
}

/// Embedding module for text-to-vector conversion
pub struct EmbeddingGenerator {
    python_path: PathBuf,
    script_path: PathBuf,
}

impl EmbeddingGenerator {
    /// Create a new EmbeddingGenerator
    pub fn new() -> Self {
        // Use conda environment Python which has our dependencies installed
        let python_path = PathBuf::from("/Users/dishankchauhan/miniconda3/bin/python");
        let script_path = PathBuf::from("python/embedding.py");
        
        EmbeddingGenerator {
            python_path,
            script_path,
        }
    }

    /// Create a new EmbeddingGenerator with custom paths
    pub fn with_paths(python_path: PathBuf, script_path: PathBuf) -> Self {
        EmbeddingGenerator {
            python_path,
            script_path,
        }
    }

    /// Convert text to a vector embedding
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        info!("Generating embedding for text: {}", text);
        
        if text.is_empty() {
            warn!("Empty text provided for embedding");
            return Ok(Vec::new());
        }
        
        // Validate that script exists
        if !self.script_path.exists() {
            error!("Embedding script not found at: {:?}", self.script_path);
            return Err(anyhow::anyhow!("Embedding script not found"));
        }
        
        // Call Python script
        let output = Command::new(&self.python_path)
            .arg(&self.script_path)
            .arg(text)
            .output()
            .context("Failed to execute embedding script")?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            error!("Embedding script error: {}", error);
            return Err(anyhow::anyhow!("Embedding script failed: {}", error));
        }
        
        // Parse output as JSON
        let output_str = String::from_utf8_lossy(&output.stdout);
        let response: EmbeddingResponse = serde_json::from_str(&output_str)
            .context("Failed to parse embedding response")?;
        
        // Check for errors from the Python script
        if let Some(error) = response.error {
            error!("Embedding generation failed: {}", error);
            return Err(anyhow::anyhow!("Embedding generation failed: {}", error));
        }
        
        info!("Generated embedding with {} dimensions", response.embedding.len());
        Ok(response.embedding)
    }
    
    /// Fallback method - returns a simple placeholder embedding
    /// Useful when the Python script is not available
    pub fn simple_embedding(&self, text: &str) -> Vec<f32> {
        warn!("Using simple fallback embedding for: {}", text);
        // This is a very naive way to create an "embedding" based on character codes
        // Only used as a fallback when the real embedding script isn't available
        let v: Vec<f32> = text.chars()
            .take(10)  // Just use the first 10 chars
            .map(|c| c as u32 as f32 / 1000.0)  // Normalize to small values
            .collect();
        
        // Pad to 10 dimensions if needed
        let mut padded = v;
        padded.resize(10, 0.0);
        padded
    }
} 