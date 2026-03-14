use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Application state
#[derive(Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: Arc<Tokenizer>,
}

// Core prediction function
pub async fn predict_sentiment(state: &AppState, text: &str) -> Result<(String, f32, Vec<f32>)> {
    // Tokenize the input
    let encoding = state
        .tokenizer
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
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .context("Failed to run model inference")?;

        // Extract logits - returns (Shape, &[f32])
        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        // Compute and return probs (copies data)
        softmax_slice(&logits_data[..2])
    };

    // Get prediction
    let predicted_idx = if probs[1] > probs[0] { 1 } else { 0 };
    let label = if predicted_idx == 1 {
        "POSITIVE"
    } else {
        "NEGATIVE"
    };
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
