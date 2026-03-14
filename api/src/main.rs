use anyhow::{Context, Result};
use api::{predict_sentiment, sentiment_analysis_v1_routes, AppState};
use axum::{routing::get, Router};
use ort::session::{builder::GraphOptimizationLevel, Session};
use parking_lot::Mutex;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Initialize ONNX Runtime and load the model
    info!("Initializing ONNX Runtime...");
    let model_path = "../models/text-classification/v1/model.onnx";
    info!("📁 Loading model from: {}", model_path);

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model. Make sure ../models/text-classification/v1/model.onnx exists!")?;

    info!("✓ Model loaded successfully!");

    // Print model info
    info!("Model inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        info!("  [{}] {} - {:?}", i, input.name, input.input_type);
    }
    info!("Model outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        info!("  [{}] {} - {:?}", i, output.name, output.output_type);
    }

    // Load the tokenizer
    let tokenizer_path = "../models/text-classification/v1/tokenizer.json";
    info!("📁 Loading tokenizer from: {}", tokenizer_path);

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    info!("✓ Tokenizer loaded successfully!");
    info!("========================================");

    // Create application state
    let state = AppState {
        session: Arc::new(Mutex::new(session)),
        tokenizer: Arc::new(tokenizer),
    };

    // Test prediction on startup
    info!("🧪 Running test prediction...");
    match predict_sentiment(&state, "This is amazing!").await {
        Ok((label, score, _)) => {
            info!(
                "✓ Test prediction successful: {} ({:.1}%)",
                label,
                score * 100.0
            );
        }
        Err(e) => {
            warn!("⚠ Test prediction failed: {}", e);
        }
    }
    info!("========================================");

    // Build the router
    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .merge(sentiment_analysis_v1_routes())
        .with_state(state);

    // Start the server
    let addr = "0.0.0.0:3000";
    info!("🚀 Server listening on http://{}", addr);
    info!("📝 Endpoints:");
    info!("   GET  /         - Health check");
    info!("   GET  /health   - Health check");
    info!("   POST /v1/predict - Single prediction");
    info!("========================================");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// Health check endpoint
async fn health_check() -> &'static str {
    "✓ Sentiment Analysis API is running!"
}


