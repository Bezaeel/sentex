use anyhow::Context;
use ort::session::{builder::GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;
use tracing::info;

pub struct LoaderResult {
    pub session: Session,
    pub tokenizer: Tokenizer,
}

pub fn load() -> Result<LoaderResult, anyhow::Error> {
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
    Ok(LoaderResult { session, tokenizer })
}
