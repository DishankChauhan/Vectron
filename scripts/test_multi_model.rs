use crate::embedding::EmbeddingGenerator;
use std::env;

fn main() {
    // Configure logging
    env_logger::init();
    
    // Get optional model ID from command line args
    let args: Vec<String> = env::args().collect();
    let model_id = args.get(1).cloned();
    
    // Text samples for testing
    let texts = vec![
        "This is a test of the multi-model embedding system.".to_string(),
        "Vector databases use embeddings for semantic search.".to_string(),
        "Different embedding models have different strengths.".to_string(),
    ];
    
    println!("Vectron Multi-Model Embedding Test");
    println!("==================================");
    
    // Create embedding generator
    let generator = match &model_id {
        Some(id) => {
            println!("Using model: {}", id);
            EmbeddingGenerator::with_model(id)
        },
        None => {
            println!("Using default model");
            EmbeddingGenerator::new()
        }
    };
    
    // List available models
    match generator.list_models() {
        Ok(models) => {
            println!("\nAvailable Models:");
            println!("Default model: {}", models.default);
            
            for (id, config) in models.models {
                println!("- {}: {} ({}d)", id, config.model_name, config.dimensions);
            }
        },
        Err(e) => {
            println!("Error listing models: {}", e);
        }
    }
    
    // Generate single embedding
    println!("\nGenerating single embedding...");
    match generator.embed_text(&texts[0]) {
        Ok(embedding) => {
            println!("Success! Generated embedding with {} dimensions", embedding.len());
            println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
        },
        Err(e) => {
            println!("Error generating embedding: {}", e);
        }
    }
    
    // Generate batch embeddings
    println!("\nGenerating batch embeddings...");
    match generator.batch_embed_texts(&texts) {
        Ok(embeddings) => {
            println!("Success! Generated {} embeddings", embeddings.len());
            
            for (i, embedding) in embeddings.iter().enumerate() {
                println!("Embedding {}: {} dimensions", i + 1, embedding.len());
                println!("  First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
            }
            
            // Calculate similarity between first two embeddings as a demonstration
            if embeddings.len() >= 2 {
                let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
                println!("\nSimilarity between first two texts: {:.4}", similarity);
            }
        },
        Err(e) => {
            println!("Error generating batch embeddings: {}", e);
        }
    }
}

// Calculate cosine similarity between two vectors
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    
    let norm_v1: f32 = v1.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let norm_v2: f32 = v2.iter().map(|&a| a * a).sum::<f32>().sqrt();
    
    dot_product / (norm_v1 * norm_v2)
} 