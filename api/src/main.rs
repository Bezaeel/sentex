use anyhow::Result;
use api::sentiment_analysis::common::loader::load as load_sentiment;
use api::{predict_sentiment, sentiment_analysis_v1_routes, AppState, SentimentState};
use axum::{routing::get, Router};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let loader_result = load_sentiment()?;
    let sentiment = SentimentState {
        session: Arc::new(Mutex::new(loader_result.session)),
        tokenizer: Arc::new(loader_result.tokenizer),
    };

    info!("Loading translation model (this may take a moment)...");
    let translation = tokio::task::spawn_blocking(api::translation::load).await??;
    info!("✓ Translation model loaded!");
    info!("========================================");

    let state = AppState {
        sentiment,
        translation,
    };

    info!("🧪 Running test prediction...");
    match predict_sentiment(&state.sentiment, "This is amazing!").await {
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

    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .merge(sentiment_analysis_v1_routes())
        .merge(api::translation::rust_bert::translate_endpoint())
        .with_state(state);

    let addr = "0.0.0.0:3000";
    info!("🚀 Server listening on http://{}", addr);
    info!("📝 Endpoints:");
    info!("   GET  /            - Health check");
    info!("   GET  /health      - Health check");
    info!("   POST /v1/predict  - Sentiment prediction");
    info!("   POST /v1/translate - Translation");
    info!("========================================");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> &'static str {
    "✓ Sentiment Analysis API is running!"
}
