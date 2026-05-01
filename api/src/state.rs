use crate::sentiment_analysis::SentimentState;
use crate::translation::TranslationState;

#[derive(Clone)]
pub struct AppState {
    pub sentiment: SentimentState,
    pub translation: TranslationState,
}
