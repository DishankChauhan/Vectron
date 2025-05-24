mod api;
mod db;
mod embedding;
mod utils;

use std::net::SocketAddr;

use dotenv::dotenv;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // Load environment variables
    dotenv().ok();
    
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "vectron=debug,tower_http=debug".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    // Build application with routes
    let app = api::router();
    
    // Set the server address
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    
    info!("Vectron Vector DB server starting on {}", addr);
    
    // Start the server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
