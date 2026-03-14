pub mod common;
pub mod v1;

pub use common::{predict_sentiment, AppState};
pub use v1::sentiment_analysis_v1_routes;
