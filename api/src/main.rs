use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{info, warn};

// Request/Response types
#[derive(Debug, Deserialize)]
struct SentimentRequest {
    text: String,
}

#[derive(Debug, Serialize)]
struct SentimentResponse {
    text: String,
    label: String,
    score: f32,
    scores: Scores,
}

#[derive(Debug, Serialize)]
struct Scores {
    negative: f32,
    positive: f32,
}

#[derive(Debug, Deserialize)]
struct BatchRequest {
    texts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BatchResponse {
    results: Vec<SentimentResponse>,
}

// Application state
#[derive(Clone)]
struct AppState {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🦀 Starting Rust Sentiment Analysis API");
    info!("========================================");

    // Initialize ONNX Runtime and load the model
    info!("Initializing ONNX Runtime...");
    let model_path = "../../models/model.onnx";
    info!("📁 Loading model from: {}", model_path);

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)
        .context("Failed to load ONNX model. Make sure ../../models/model.onnx exists!")?;

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
    let tokenizer_path = "../../models/tokenizer.json";
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
            info!("✓ Test prediction successful: {} ({:.1}%)", label, score * 100.0);
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
        .route("/predict", post(predict_handler))
        .route("/batch", post(batch_predict_handler))
        .with_state(state);

    // Start the server
    let addr = "0.0.0.0:3000";
    info!("🚀 Server listening on http://{}", addr);
    info!("📝 Endpoints:");
    info!("   GET  /        - Health check");
    info!("   GET  /health  - Health check");
    info!("   POST /predict - Single prediction");
    info!("   POST /batch   - Batch predictions");
    info!("========================================");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// Health check endpoint
async fn health_check() -> &'static str {
    "✓ Sentiment Analysis API is running!"
}

// Single prediction endpoint
async fn predict_handler(
    State(state): State<AppState>,
    Json(request): Json<SentimentRequest>,
) -> Result<Json<SentimentResponse>, (StatusCode, String)> {
    let text = request.text.trim();

    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Text cannot be empty".to_string()));
    }

    match predict_sentiment(&state, text).await {
        Ok((label, score, scores)) => {
            Ok(Json(SentimentResponse {
                text: text.to_string(),
                label,
                score,
                scores: Scores {
                    negative: scores[0],
                    positive: scores[1],
                },
            }))
        }
        Err(e) => {
            warn!("Prediction error: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Prediction failed: {}", e)))
        }
    }
}

// Batch prediction endpoint
async fn batch_predict_handler(
    State(state): State<AppState>,
    Json(request): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, (StatusCode, String)> {
    if request.texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Texts array cannot be empty".to_string()));
    }

    if request.texts.len() > 100 {
        return Err((StatusCode::BAD_REQUEST, "Maximum 100 texts per batch".to_string()));
    }

    let mut results = Vec::new();

    for text in request.texts {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        match predict_sentiment(&state, text).await {
            Ok((label, score, scores)) => {
                results.push(SentimentResponse {
                    text: text.to_string(),
                    label,
                    score,
                    scores: Scores {
                        negative: scores[0],
                        positive: scores[1],
                    },
                });
            }
            Err(e) => {
                warn!("Prediction error for text '{}': {}", text, e);
                // Continue with other predictions
            }
        }
    }

    Ok(Json(BatchResponse { results }))
}

// Core prediction function
async fn predict_sentiment(
    state: &AppState,
    text: &str,
) -> Result<(String, f32, Vec<f32>)> {
    // Tokenize the input
    let encoding = state.tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize input: {}", e))?;

    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    // Convert to i64 and create arrays
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

    let seq_len = input_ids_i64.len();

    // Create input tensors
    let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids_i64)
        .context("Failed to create input_ids array")?;

    let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask_i64)
        .context("Failed to create attention_mask array")?;

    // Create ONNX tensors
    let input_ids_tensor = Tensor::from_array(input_ids_array)?;
    let attention_mask_tensor = Tensor::from_array(attention_mask_array)?;

    // Run inference and extract probabilities within the lock scope
    let probs = {
        let mut session = state.session.lock();
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])
        .context("Failed to run model inference")?;

        // Extract logits
        let logits_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        let logits_data = logits_tensor.1;

        // Compute and return probs (copies data)
        softmax_slice(&logits_data[..2])
    };

    // Get prediction
    let predicted_idx = if probs[1] > probs[0] { 1 } else { 0 };
    let label = if predicted_idx == 1 { "POSITIVE" } else { "NEGATIVE" };
    let score = probs[predicted_idx];

    Ok((label.to_string(), score, probs))
}

// Softmax function
fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}
