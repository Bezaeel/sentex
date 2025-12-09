use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug)]
struct Review {
    text: String,
    actual_label: i32, // 0 = negative, 1 = positive
}

#[derive(Debug)]
struct Prediction {
    text: String,
    actual_label: String,
    predicted_label: String,
    confidence: f32,
    correct: bool,
}

#[derive(Debug)]
struct Metrics {
    total: usize,
    correct: usize,
    accuracy: f32,
    true_positives: usize,
    true_negatives: usize,
    false_positives: usize,
    false_negatives: usize,
    precision: f32,
    recall: f32,
    f1_score: f32,
}

fn main() -> Result<()> {
    println!("========================================");
    println!("🧪 Testing ONNX Model on Yelp Dataset");
    println!("========================================\n");

    // Load the Yelp dataset
    println!("📂 Loading Yelp dataset...");
    let reviews = load_yelp_dataset("yelp_labelled.txt")?;
    println!("   Loaded {} reviews\n", reviews.len());

    // Initialize ONNX Runtime and load model
    println!("🔧 Initializing ONNX Runtime...");
    println!("📁 Loading ONNX model...");
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../../models/model.onnx")
        .context("Failed to load model. Make sure models/model.onnx exists!")?;
    println!("   ✓ Model loaded\n");

    // Load tokenizer
    println!("📁 Loading tokenizer...");
    let tokenizer = Tokenizer::from_file("../../models/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("   ✓ Tokenizer loaded\n");

    // Run predictions
    println!("🔮 Running predictions on {} reviews...", reviews.len());
    let start_time = Instant::now();

    let mut predictions = Vec::new();
    let mut errors = 0;
    let mut session = session;

    for (i, review) in reviews.iter().enumerate() {
        if (i + 1) % 50 == 0 {
            println!("   Progress: {}/{}", i + 1, reviews.len());
        }

        match predict_sentiment(&mut session, &tokenizer, &review.text) {
            Ok((predicted_idx, confidence, _)) => {
                let predicted_label = if predicted_idx == 1 {
                    "POSITIVE"
                } else {
                    "NEGATIVE"
                };
                let actual_label = if review.actual_label == 1 {
                    "POSITIVE"
                } else {
                    "NEGATIVE"
                };
                let correct = predicted_idx == review.actual_label;

                predictions.push(Prediction {
                    text: review.text.clone(),
                    actual_label: actual_label.to_string(),
                    predicted_label: predicted_label.to_string(),
                    confidence,
                    correct,
                });
            }
            Err(e) => {
                eprintln!(
                    "   ✗ Error predicting '{}': {}",
                    &review.text[..50.min(review.text.len())],
                    e
                );
                errors += 1;
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!("   ✓ Completed in {:.2}s", elapsed.as_secs_f64());
    println!(
        "   Average: {:.2}ms per prediction\n",
        elapsed.as_millis() as f64 / reviews.len() as f64
    );

    if errors > 0 {
        println!("   ⚠ {} errors encountered\n", errors);
    }

    // Calculate metrics
    let metrics = calculate_metrics(&predictions);

    // Print results
    println!("========================================");
    println!("📊 Results");
    println!("========================================\n");

    println!("Overall Performance:");
    println!("  Total Reviews:    {}", metrics.total);
    println!("  Correct:          {}", metrics.correct);
    println!("  Incorrect:        {}", metrics.total - metrics.correct);
    println!("  Accuracy:         {:.2}%\n", metrics.accuracy * 100.0);

    println!("Confusion Matrix:");
    println!("                 Predicted Positive  Predicted Negative");
    println!(
        "  Actual Positive      {:4}                {:4}",
        metrics.true_positives, metrics.false_negatives
    );
    println!(
        "  Actual Negative      {:4}                {:4}\n",
        metrics.false_positives, metrics.true_negatives
    );

    println!("Performance Metrics:");
    println!("  Precision:        {:.2}%", metrics.precision * 100.0);
    println!("  Recall:           {:.2}%", metrics.recall * 100.0);
    println!("  F1 Score:         {:.2}%\n", metrics.f1_score * 100.0);

    // Show some example predictions
    println!("========================================");
    println!("📝 Sample Predictions");
    println!("========================================\n");

    println!("✓ Correct Predictions (5 examples):");
    for pred in predictions.iter().filter(|p| p.correct).take(5) {
        println!("  Text: \"{}\"", &pred.text[..60.min(pred.text.len())]);
        println!(
            "  Actual: {} | Predicted: {} | Confidence: {:.1}%\n",
            pred.actual_label,
            pred.predicted_label,
            pred.confidence * 100.0
        );
    }

    println!("✗ Incorrect Predictions (5 examples):");
    for pred in predictions.iter().filter(|p| !p.correct).take(5) {
        println!("  Text: \"{}\"", &pred.text[..60.min(pred.text.len())]);
        println!(
            "  Actual: {} | Predicted: {} | Confidence: {:.1}%\n",
            pred.actual_label,
            pred.predicted_label,
            pred.confidence * 100.0
        );
    }

    // Low confidence predictions
    let mut low_confidence: Vec<_> = predictions.iter().filter(|p| p.confidence < 0.7).collect();
    low_confidence.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

    if !low_confidence.is_empty() {
        println!(
            "⚠ Low Confidence Predictions ({} total):",
            low_confidence.len()
        );
        for pred in low_confidence.iter().take(5) {
            println!("  Text: \"{}\"", &pred.text[..60.min(pred.text.len())]);
            println!(
                "  Predicted: {} | Confidence: {:.1}% | Correct: {}\n",
                pred.predicted_label,
                pred.confidence * 100.0,
                pred.correct
            );
        }
    }

    println!("========================================");
    println!("✅ Testing Complete!");
    println!("========================================");

    Ok(())
}

fn load_yelp_dataset(path: &str) -> Result<Vec<Review>> {
    let file = File::open(path).context(format!(
        "Failed to open {}. Make sure it's in the current directory!",
        path
    ))?;
    let reader = BufReader::new(file);

    let mut reviews = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();

        if parts.len() == 2 {
            let text = parts[0].trim().to_string();
            let label = parts[1].trim().parse::<i32>()?;

            if !text.is_empty() {
                reviews.push(Review {
                    text,
                    actual_label: label,
                });
            }
        }
    }

    Ok(reviews)
}

fn predict_sentiment(
    session: &mut Session,
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<(i32, f32, Vec<f32>)> {
    // Tokenize
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    // Convert to i64
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
    let seq_len = input_ids_i64.len();

    // Create tensors
    let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids_i64)?;
    let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask_i64)?;

    // Run inference
    let input_ids_tensor = Tensor::from_array(input_ids_array)?;
    let attention_mask_tensor = Tensor::from_array(attention_mask_array)?;

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    ])?;

    // Extract logits
    let logits_tensor = outputs[0].try_extract_tensor::<f32>()?;
    let logits_data = logits_tensor.1;
    let probs = softmax_slice(&logits_data[..2]);

    let predicted_idx: i32 = if probs[1] > probs[0] { 1 } else { 0 };
    let confidence = probs[predicted_idx as usize];

    Ok((predicted_idx, confidence, probs))
}

fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

fn calculate_metrics(predictions: &[Prediction]) -> Metrics {
    let total = predictions.len();
    let correct = predictions.iter().filter(|p| p.correct).count();

    let true_positives = predictions
        .iter()
        .filter(|p| p.actual_label == "POSITIVE" && p.predicted_label == "POSITIVE")
        .count();

    let true_negatives = predictions
        .iter()
        .filter(|p| p.actual_label == "NEGATIVE" && p.predicted_label == "NEGATIVE")
        .count();

    let false_positives = predictions
        .iter()
        .filter(|p| p.actual_label == "NEGATIVE" && p.predicted_label == "POSITIVE")
        .count();

    let false_negatives = predictions
        .iter()
        .filter(|p| p.actual_label == "POSITIVE" && p.predicted_label == "NEGATIVE")
        .count();

    let accuracy = correct as f32 / total as f32;

    let precision = if (true_positives + false_positives) > 0 {
        true_positives as f32 / (true_positives + false_positives) as f32
    } else {
        0.0
    };

    let recall = if (true_positives + false_negatives) > 0 {
        true_positives as f32 / (true_positives + false_negatives) as f32
    } else {
        0.0
    };

    let f1_score = if (precision + recall) > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    Metrics {
        total,
        correct,
        accuracy,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1_score,
    }
}
