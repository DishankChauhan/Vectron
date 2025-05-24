use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use tracing::{info, warn};

use crate::utils::cosine_similarity;

// Submodules
pub mod persistence;

/// Result type for similarity search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score (higher is more similar)
    pub score: f32,
}

/// VectorStore represents our in-memory storage of vectors
#[derive(Debug)]
pub struct VectorStore {
    // Map from string ID to vector of f32 values
    vectors: HashMap<String, Vec<f32>>,
    // Persistence manager for saving/loading data
    persistence: Option<persistence::PersistenceManager>,
    // Whether to automatically save on changes
    auto_save: bool,
}

impl Default for VectorStore {
    fn default() -> Self {
        Self {
            vectors: HashMap::new(),
            persistence: None,
            auto_save: false,
        }
    }
}

impl VectorStore {
    /// Create a new in-memory vector store
    pub fn new() -> Self {
        VectorStore {
            vectors: HashMap::new(),
            persistence: None,
            auto_save: false,
        }
    }
    
    /// Create a new vector store with persistence
    pub fn with_persistence(storage_dir: impl AsRef<Path>, auto_save: bool) -> Result<Self> {
        let persistence = persistence::PersistenceManager::new(storage_dir);
        persistence.init()?;
        
        // Load existing vectors if available
        let vectors = persistence.load()?;
        info!("Loaded {} vectors from disk", vectors.len());
        
        Ok(VectorStore {
            vectors,
            persistence: Some(persistence),
            auto_save,
        })
    }
    
    /// Save vectors to disk (only if persistence is enabled)
    pub fn save(&self) -> Result<()> {
        if let Some(persistence) = &self.persistence {
            persistence.save(&self.vectors)?;
        } else {
            warn!("Persistence not enabled, vectors not saved");
        }
        Ok(())
    }
    
    /// Insert a vector with the given ID
    /// If a vector with this ID already exists, it will be overwritten
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> bool {
        let is_update = self.vectors.contains_key(&id);
        self.vectors.insert(id, vector);
        
        // Auto-save if enabled
        if self.auto_save {
            if let Err(e) = self.save() {
                warn!("Failed to auto-save after insert: {:?}", e);
            }
        }
        
        is_update
    }
    
    /// Get a vector by ID
    /// Returns None if the vector doesn't exist
    pub fn get(&self, id: &str) -> Option<&Vec<f32>> {
        self.vectors.get(id)
    }
    
    /// Delete a vector by ID
    /// Returns true if the vector was deleted, false if it didn't exist
    pub fn delete(&mut self, id: &str) -> bool {
        let result = self.vectors.remove(id).is_some();
        
        // Auto-save if enabled
        if result && self.auto_save {
            if let Err(e) = self.save() {
                warn!("Failed to auto-save after delete: {:?}", e);
            }
        }
        
        result
    }
    
    /// Get the number of vectors in the store
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    
    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    
    /// List all vector IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }
    
    /// Search for similar vectors
    /// Returns a list of vector IDs and similarity scores, sorted by similarity
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self.vectors
            .iter()
            .map(|(id, vector)| {
                let score = cosine_similarity(query, vector);
                SearchResult {
                    id: id.clone(),
                    score,
                }
            })
            .collect();
        
        // Sort by score in descending order (higher is more similar)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k results
        if results.len() > top_k {
            results.truncate(top_k);
        }
        
        results
    }
}

// Shared storage that can be safely accessed from multiple threads
pub type SharedVectorStore = Arc<RwLock<VectorStore>>; 