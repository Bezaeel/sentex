pub mod rust_bert;
pub mod burn;

use std::sync::Arc;
use ::rust_bert::pipelines::translation::TranslationModel;
use parking_lot::Mutex;
// re-export so can be used in main.rs as translation::load()
pub use rust_bert::load;

#[derive(Clone)]
pub struct TranslationState {
    pub model: Arc<Mutex<TranslationModel>>,
}
