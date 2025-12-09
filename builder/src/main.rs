use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // Load ONNX model
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("models/model.onnx")?;

    // Load tokenizer
    let tokenizer =
        Tokenizer::from_file("models/tokenizer.json").map_err(|e| anyhow::anyhow!("{}", e))?;

    // Tokenize input
    let text = "This movie is amazing!";
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Prepare inputs
    let seq_len = encoding.get_ids().len();
    let input_ids: Array2<i64> = Array2::from_shape_vec(
        (1, seq_len),
        encoding.get_ids().iter().map(|&x| x as i64).collect(),
    )?;

    let attention_mask: Array2<i64> = Array2::from_shape_vec(
        (1, seq_len),
        encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect(),
    )?;

    // Run inference
    let input_ids_tensor = Tensor::from_array(input_ids)?;
    let attention_mask_tensor = Tensor::from_array(attention_mask)?;

    let outputs = model.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    ])?;

    // Process results
    let logits = outputs[0].try_extract_tensor::<f32>()?;
    let (_, logits_data) = logits;
    let probs = softmax(logits_data);

    println!(
        "Negative: {:.2}%, Positive: {:.2}%",
        probs[0] * 100.0,
        probs[1] * 100.0
    );

    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}
