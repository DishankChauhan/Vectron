use std::process::Command;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{info, warn, error};
use reqwest::Client;
use std::env;
use std::time::Duration;

/// Response format from Python embedding script
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingResponse {
    pub text: Option<String>,
    pub dimensions: Option<usize>,
    pub embedding: Vec<f32>,
    pub error: Option<String>,
}

/// Model info returned from the embedding service
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub dimensions: i32,
    pub is_remote: bool,
    pub r#type: String,
}

/// Response format from embedding service
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingServiceResponse {
    pub embedding: Vec<f32>,
    pub dimensions: i32,
    pub text: String,
    pub processing_time_ms: f32,
    pub model_info: ModelInfo,
}

/// Response format from batch embedding service
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchEmbeddingServiceResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dimensions: i32,
    pub count: i32,
    pub processing_time_ms: f32,
    pub model_info: ModelInfo,
}

/// Response format from models endpoint
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelsResponse {
    pub default: String,
    pub models: std::collections::HashMap<String, ModelConfig>,
}

/// Model configuration format
#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub r#type: String,
    pub model_name: String,
    pub dimensions: i32,
}

/// Embedding module for text-to-vector conversion
pub struct EmbeddingGenerator {
    python_path: PathBuf,
    script_path: PathBuf,
    http_client: Client,
    service_url: String,
    use_service: bool,
    model_id: Option<String>,
}

impl EmbeddingGenerator {
    /// Create a new EmbeddingGenerator
    pub fn new() -> Self {
        // Check if we should use the embedding service
        let service_url = env::var("EMBEDDING_SERVICE_URL")
            .unwrap_or_else(|_| "http://localhost:8000".to_string());
        
        // Get optional model_id from environment
        let model_id = env::var("EMBEDDING_MODEL")
            .ok();
        
        // Use conda environment Python which has our dependencies installed
        let python_path = PathBuf::from("/Users/dishankchauhan/miniconda3/bin/python");
        let script_path = PathBuf::from("python/embedding.py");
        
        // Create HTTP client for service requests
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        
        let use_service = !service_url.is_empty();
        
        if use_service {
            if let Some(model) = &model_id {
                info!("Using embedding service at: {} with model: {}", service_url, model);
            } else {
                info!("Using embedding service at: {} with default model", service_url);
            }
        } else {
            info!("Using local Python script for embeddings");
        }
        
        EmbeddingGenerator {
            python_path,
            script_path,
            http_client,
            service_url,
            use_service,
            model_id,
        }
    }

    /// Create a new EmbeddingGenerator with custom paths
    pub fn with_paths(python_path: PathBuf, script_path: PathBuf) -> Self {
        // Check if we should use the embedding service
        let service_url = env::var("EMBEDDING_SERVICE_URL")
            .unwrap_or_else(|_| "http://localhost:8000".to_string());
        
        // Get optional model_id from environment
        let model_id = env::var("EMBEDDING_MODEL")
            .ok();
            
        // Create HTTP client for service requests
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
            
        let use_service = !service_url.is_empty();
        
        EmbeddingGenerator {
            python_path,
            script_path,
            http_client,
            service_url,
            use_service,
            model_id,
        }
    }
    
    /// Create a new EmbeddingGenerator with a specific model
    pub fn with_model(model_id: &str) -> Self {
        let mut generator = Self::new();
        generator.model_id = Some(model_id.to_string());
        generator
    }

    /// Get the list of available models from the service
    pub fn list_models(&self) -> Result<ModelsResponse> {
        if !self.use_service {
            return Err(anyhow::anyhow!("Embedding service not configured"));
        }
        
        // Make a blocking request to get models
        let response = reqwest::blocking::Client::new()
            .get(&format!("{}/models", self.service_url))
            .send()
            .context("Failed to call embedding service")?;
            
        if !response.status().is_success() {
            let error = response.text()?;
            error!("Embedding service error: {}", error);
            return Err(anyhow::anyhow!("Failed to get models: {}", error));
        }
        
        // Parse response
        let models: ModelsResponse = response
            .json()
            .context("Failed to parse models response")?;
        
        Ok(models)
    }

    /// Convert text to a vector embedding using the service
    async fn embed_text_service(&self, text: &str) -> Result<Vec<f32>> {
        info!("Generating embedding via service for text: {}", text);
        
        if text.is_empty() {
            warn!("Empty text provided for embedding");
            return Ok(Vec::new());
        }
        
        // Build the request URL with optional model parameter
        let mut url = format!("{}/embed", self.service_url);
        if let Some(model) = &self.model_id {
            url = format!("{}?model_id={}", url, model);
        }
        
        // Call the embedding service
        let response = self.http_client
            .post(&url)
            .json(&serde_json::json!({ "text": text }))
            .send()
            .await
            .context("Failed to call embedding service")?;
            
        if !response.status().is_success() {
            let error = response.text().await?;
            error!("Embedding service error: {}", error);
            return Err(anyhow::anyhow!("Embedding service failed: {}", error));
        }
        
        // Parse response
        let service_response: EmbeddingServiceResponse = response
            .json()
            .await
            .context("Failed to parse embedding service response")?;
        
        info!(
            "Generated embedding with {} dimensions using model: {}",
            service_response.dimensions,
            service_response.model_info.name
        );
        
        Ok(service_response.embedding)
    }
    
    /// Convert multiple texts to vector embeddings using the service
    pub fn batch_embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if !self.use_service {
            // If service is not available, embed each text individually
            return Ok(texts
                .iter()
                .map(|text| self.embed_text(text).unwrap_or_else(|_| self.simple_embedding(text)))
                .collect());
        }
        
        info!("Generating batch embeddings via service for {} texts", texts.len());
        
        if texts.is_empty() {
            warn!("Empty batch provided for embedding");
            return Ok(Vec::new());
        }
        
        // Build the request URL with optional model parameter
        let mut url = format!("{}/batch-embed", self.service_url);
        if let Some(model) = &self.model_id {
            url = format!("{}?model_id={}", url, model);
        }
        
        // Make a blocking request to the embedding service
        let response = reqwest::blocking::Client::new()
            .post(&url)
            .json(&serde_json::json!({ "texts": texts }))
            .send()
            .context("Failed to call embedding service for batch")?;
            
        if !response.status().is_success() {
            let error = response.text()?;
            error!("Embedding service batch error: {}", error);
            
            // Fall back to individual embedding
            warn!("Falling back to individual embedding");
            return Ok(texts
                .iter()
                .map(|text| self.embed_text(text).unwrap_or_else(|_| self.simple_embedding(text)))
                .collect());
        }
        
        // Parse response
        let service_response: BatchEmbeddingServiceResponse = response
            .json()
            .context("Failed to parse batch embedding service response")?;
        
        info!(
            "Generated {} embeddings with {} dimensions using model: {}", 
            service_response.count,
            service_response.dimensions,
            service_response.model_info.name
        );
        
        Ok(service_response.embeddings)
    }

    /// Convert text to a vector embedding
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // If we're configured to use the service, try that first
        if self.use_service {
            // Since we can't use async in a sync function, we'll use a blocking client
            
            info!("Generating embedding via service for text: {}", text);
            
            if text.is_empty() {
                warn!("Empty text provided for embedding");
                return Ok(Vec::new());
            }
            
            // Build the request URL with optional model parameter
            let mut url = format!("{}/embed", self.service_url);
            if let Some(model) = &self.model_id {
                url = format!("{}?model_id={}", url, model);
            }
            
            // Make a blocking request to the embedding service
            let response = reqwest::blocking::Client::new()
                .post(&url)
                .json(&serde_json::json!({ "text": text }))
                .send()
                .context("Failed to call embedding service")?;
                
            if !response.status().is_success() {
                let error = response.text()?;
                error!("Embedding service error: {}", error);
                
                // Fall back to local Python script
                warn!("Falling back to local Python script");
                return self.embed_text_local(text);
            }
            
            // Parse response
            let service_response: EmbeddingServiceResponse = response
                .json()
                .context("Failed to parse embedding service response")?;
            
            info!(
                "Generated embedding with {} dimensions using model: {}",
                service_response.dimensions,
                service_response.model_info.name
            );
            
            return Ok(service_response.embedding);
        }
        
        // If we're not using the service or it failed, use the local Python script
        self.embed_text_local(text)
    }
    
    /// Convert text to a vector embedding using local Python script
    fn embed_text_local(&self, text: &str) -> Result<Vec<f32>> {
        info!("Generating embedding via Python script for text: {}", text);
        
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