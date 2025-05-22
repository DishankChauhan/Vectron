use std::sync::{Arc, RwLock};

use axum::{
    extract::{Path, State, Query},
    routing::{get, post, delete},
    Router, Json,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::db::VectorStore;
use crate::embedding::EmbeddingGenerator;

// Submodules
pub mod benchmark;

// Type alias for shared state
type AppState = Arc<AppStateInner>;

// Application state
pub struct AppStateInner {
    store: RwLock<VectorStore>,
    embedding_generator: EmbeddingGenerator,
}

// Request and response structs
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorData {
    id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorResponse {
    id: String,
    vector: Option<Vec<f32>>,
    success: bool,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    id: String,
    score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    results: Vec<SearchResult>,
    query: String,
    success: bool,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    success: bool,
    message: String,
    count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextData {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorSearchQuery {
    vector: Vec<f32>,
    top_k: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextSearchQuery {
    text: String,
    top_k: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextEmbeddingResponse {
    text: String,
    embedding: Vec<f32>,
    dimensions: usize,
    success: bool,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextVectorData {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    top_k: Option<usize>,
}

pub fn router() -> Router {
    let embedding_generator = EmbeddingGenerator::new();
    
    // Create a persistence-enabled vector store 
    let store = match VectorStore::with_persistence("./data", true) {
        Ok(store) => store,
        Err(e) => {
            warn!("Failed to initialize persistence: {}, falling back to in-memory store", e);
            VectorStore::new()
        }
    };
    
    let state = Arc::new(AppStateInner {
        store: RwLock::new(store),
        embedding_generator,
    });
    
    Router::new()
        .route("/", get(health_check))
        .route("/vector", post(insert_vector))
        .route("/vector/:id", get(get_vector))
        .route("/vector/:id", delete(delete_vector))
        .route("/vectors", get(list_vectors))
        .route("/embed", post(embed_text))
        .route("/upsert-text", post(upsert_text))
        .route("/search/vector", post(search_by_vector))
        .route("/search/text", post(search_by_text))
        .route("/benchmark", post(benchmark::run_benchmark))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "Hello from VectorDB!"
}

// Insert or update a vector
async fn insert_vector(
    State(state): State<AppState>,
    Json(data): Json<VectorData>
) -> Json<StatusResponse> {
    info!("Inserting vector with ID: {}", data.id);
    
    let is_update = {
        let mut store = state.store.write().unwrap();
        store.insert(data.id.clone(), data.vector)
    };
    
    let message = if is_update {
        format!("Vector with ID {} updated", data.id)
    } else {
        format!("Vector with ID {} inserted", data.id)
    };
    
    let count = state.store.read().unwrap().len();
    
    Json(StatusResponse {
        success: true,
        message,
        count,
    })
}

// Get a vector by ID
async fn get_vector(
    State(state): State<AppState>,
    Path(id): Path<String>
) -> Json<VectorResponse> {
    info!("Getting vector with ID: {}", id);
    
    let result = {
        let store = state.store.read().unwrap();
        store.get(&id).cloned()
    };
    
    let (success, message, vector) = match result {
        Some(vector) => (true, format!("Vector with ID {} found", id), Some(vector)),
        None => (false, format!("Vector with ID {} not found", id), None),
    };
    
    Json(VectorResponse {
        id,
        vector,
        success,
        message,
    })
}

// Delete a vector by ID
async fn delete_vector(
    State(state): State<AppState>,
    Path(id): Path<String>
) -> Json<StatusResponse> {
    info!("Deleting vector with ID: {}", id);
    
    let success = {
        let mut store = state.store.write().unwrap();
        store.delete(&id)
    };
    
    let message = if success {
        format!("Vector with ID {} deleted", id)
    } else {
        format!("Vector with ID {} not found", id)
    };
    
    let count = state.store.read().unwrap().len();
    
    Json(StatusResponse {
        success,
        message,
        count,
    })
}

// List all vector IDs
async fn list_vectors(
    State(state): State<AppState>
) -> Json<Vec<String>> {
    info!("Listing all vector IDs");
    
    let ids = {
        let store = state.store.read().unwrap();
        store.list_ids()
    };
    
    Json(ids)
}

// Generate an embedding from text
async fn embed_text(
    State(state): State<AppState>,
    Json(text_data): Json<TextData>
) -> Json<TextEmbeddingResponse> {
    info!("Generating embedding for text");
    
    let text = text_data.text.clone();
    let result = state.embedding_generator.embed_text(&text);
    
    match result {
        Ok(embedding) => {
            let dimensions = embedding.len();
            Json(TextEmbeddingResponse {
                text,
                embedding,
                dimensions,
                success: true,
                message: format!("Generated embedding with {} dimensions", dimensions),
            })
        },
        Err(e) => {
            warn!("Error generating embedding: {}", e);
            // Use the fallback embedding method
            let embedding = state.embedding_generator.simple_embedding(&text);
            let dimensions = embedding.len();
            Json(TextEmbeddingResponse {
                text,
                embedding,
                dimensions,
                success: false,
                message: format!("Error: {}. Used fallback embedding.", e),
            })
        }
    }
}

// Create a vector from text (combine embedding and storage)
async fn upsert_text(
    State(state): State<AppState>,
    Json(data): Json<TextVectorData>
) -> Json<StatusResponse> {
    info!("Inserting vector from text with ID: {}", data.id);
    
    let text = data.text.clone();
    let result = state.embedding_generator.embed_text(&text);
    
    let embedding = match result {
        Ok(emb) => emb,
        Err(e) => {
            warn!("Error generating embedding: {}", e);
            // Use the fallback embedding method
            state.embedding_generator.simple_embedding(&text)
        }
    };
    
    let is_update = {
        let mut store = state.store.write().unwrap();
        store.insert(data.id.clone(), embedding)
    };
    
    let message = if is_update {
        format!("Vector from text with ID {} updated", data.id)
    } else {
        format!("Vector from text with ID {} inserted", data.id)
    };
    
    let count = state.store.read().unwrap().len();
    
    Json(StatusResponse {
        success: true,
        message,
        count,
    })
}

// Search by vector
async fn search_by_vector(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
    Json(query): Json<VectorSearchQuery>,
) -> Json<Vec<SearchResult>> {
    let top_k = params.top_k.unwrap_or(5);
    info!("Searching for similar vectors to input vector, top_k: {}", top_k);
    
    let results = {
        let store = state.store.read().unwrap();
        store.search(&query.vector, top_k)
    };
    
    // Convert DB results to API results
    let api_results = results.into_iter()
        .map(|result| SearchResult {
            id: result.id,
            score: result.score,
        })
        .collect();
    
    Json(api_results)
}

// Search by text
async fn search_by_text(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
    Json(query): Json<TextData>,
) -> Json<SearchResponse> {
    let top_k = params.top_k.unwrap_or(5);
    let text = query.text.clone();
    info!("Searching for similar vectors to text: '{}', top_k: {}", text, top_k);
    
    // Generate embedding for the query text
    let result = state.embedding_generator.embed_text(&text);
    
    let query_vector = match result {
        Ok(embedding) => embedding,
        Err(e) => {
            warn!("Error generating embedding for search query: {}", e);
            // Use the fallback embedding method
            state.embedding_generator.simple_embedding(&text)
        }
    };
    
    // Search for similar vectors
    let results = {
        let store = state.store.read().unwrap();
        store.search(&query_vector, top_k)
    };
    
    // Convert DB results to API results
    let api_results: Vec<SearchResult> = results.into_iter()
        .map(|result| SearchResult {
            id: result.id,
            score: result.score,
        })
        .collect();
    
    let result_count = api_results.len();
    
    Json(SearchResponse {
        results: api_results,
        query: text,
        success: true,
        message: format!("Found {} similar vectors", result_count),
    })
} 