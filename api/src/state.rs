use crate::sentiment_analysis::SentimentState;
use crate::translation::rust_bert::TranslationState;

#[derive(Clone)]
pub struct AppState {
    pub sentiment: SentimentState,
    pub translation: TranslationState,
}
