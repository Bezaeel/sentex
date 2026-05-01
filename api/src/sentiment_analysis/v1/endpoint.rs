use crate::sentiment_analysis::common::predict_sentiment;
use crate::sentiment_analysis::v1::{Scores, SentimentRequest, SentimentResponse};
use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use tracing::warn;

pub fn sentiment_analysis_v1_routes() -> Router<AppState> {
    Router::new().route("/v1/predict", post(predict_handler))
}

async fn predict_handler(
    State(state): State<AppState>,
    Json(request): Json<SentimentRequest>,
) -> Result<Json<SentimentResponse>, (StatusCode, String)> {
    let text = request.text.trim();

    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Text cannot be empty".to_string()));
    }

    match predict_sentiment(&state.sentiment, text).await {
        Ok((label, score, scores)) => Ok(Json(SentimentResponse {
            text: text.to_string(),
            label,
            score,
            scores: Scores {
                negative: scores[0],
                positive: scores[1],
            },
        })),
        Err(e) => {
            warn!("Prediction error: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Prediction failed: {}", e),
            ))
        }
    }
}
