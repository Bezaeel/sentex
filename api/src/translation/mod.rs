use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use parking_lot::Mutex;
use rust_bert::pipelines::translation::{Language, TranslationModel, TranslationModelBuilder};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Clone)]
pub struct TranslationState {
    pub model: Arc<Mutex<TranslationModel>>,
}

pub fn load() -> Result<TranslationState, anyhow::Error> {
    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![Language::Spanish])
        .with_target_languages(vec![Language::English])
        .create_model()?;
    Ok(TranslationState {
        model: Arc::new(Mutex::new(model)),
    })
}

pub fn translate_endpoint() -> Router<AppState> {
    Router::new().route("/v1/translate", post(handler))
}

#[derive(Deserialize)]
pub struct TranslateRequest {
    text: String,
}

async fn handler(
    State(state): State<AppState>,
    Json(req): Json<TranslateRequest>,
) -> Result<Json<String>, (StatusCode, String)> {
    let model = Arc::clone(&state.translation.model);
    tokio::task::spawn_blocking(move || {
        let model = model.lock();
        let output = model.translate(&[&req.text], None, Language::English)?;
        Ok::<_, anyhow::Error>(output.into_iter().next().unwrap_or_default())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("task panicked: {e}")))?
    .map(Json)
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Translation failed: {e}")))
}
