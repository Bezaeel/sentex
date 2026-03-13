use serde::{Deserialize, Serialize};

// Request/Response types
#[derive(Debug, Deserialize)]
pub struct SentimentRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct SentimentResponse {
    pub text: String,
    pub label: String,
    pub score: f32,
    pub scores: Scores,
}

#[derive(Debug, Serialize)]
pub struct Scores {
    pub negative: f32,
    pub positive: f32,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub texts: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub results: Vec<SentimentResponse>,
}
