use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Vector data ready for serialization
#[derive(Serialize, Deserialize)]
pub struct VectorData {
    pub id: String,
    pub vector: Vec<f32>,
}

/// Storage manager for persisting vectors to disk
#[derive(Debug)]
pub struct PersistenceManager {
    /// Directory where vectors are stored
    storage_dir: PathBuf,
    /// Main index file
    index_file: PathBuf,
}

impl PersistenceManager {
    /// Create a new persistence manager
    pub fn new(storage_dir: impl AsRef<Path>) -> Self {
        let storage_dir = storage_dir.as_ref().to_path_buf();
        let index_file = storage_dir.join("index.json");
        
        PersistenceManager {
            storage_dir,
            index_file,
        }
    }
    
    /// Initialize the storage directory
    pub fn init(&self) -> Result<()> {
        if !self.storage_dir.exists() {
            fs::create_dir_all(&self.storage_dir)
                .context("Failed to create storage directory")?;
            info!("Created storage directory: {:?}", self.storage_dir);
        }
        
        Ok(())
    }
    
    /// Save vectors to disk
    pub fn save(&self, vectors: &HashMap<String, Vec<f32>>) -> Result<()> {
        info!("Saving {} vectors to disk", vectors.len());
        self.init()?;
        
        // Open the index file
        let file = File::create(&self.index_file)
            .context("Failed to create index file")?;
        let mut writer = BufWriter::new(file);
        
        // Serialize the vectors
        let vector_data: Vec<VectorData> = vectors
            .iter()
            .map(|(id, vector)| VectorData {
                id: id.clone(),
                vector: vector.clone(),
            })
            .collect();
        
        // Write to the file
        serde_json::to_writer(&mut writer, &vector_data)
            .context("Failed to write vectors to index file")?;
        
        writer.flush().context("Failed to flush index file")?;
        info!("Successfully saved vectors to disk");
        
        Ok(())
    }
    
    /// Load vectors from disk
    pub fn load(&self) -> Result<HashMap<String, Vec<f32>>> {
        if !self.index_file.exists() {
            info!("No index file found, starting with empty database");
            return Ok(HashMap::new());
        }
        
        info!("Loading vectors from disk");
        
        // Open the index file
        let file = File::open(&self.index_file)
            .context("Failed to open index file")?;
        let reader = BufReader::new(file);
        
        // Deserialize the vectors
        let vector_data: Vec<VectorData> = serde_json::from_reader(reader)
            .context("Failed to read vectors from index file")?;
        
        // Convert to HashMap
        let vectors = vector_data
            .into_iter()
            .map(|data| (data.id, data.vector))
            .collect();
        
        info!("Successfully loaded vectors from disk");
        
        Ok(vectors)
    }
} 