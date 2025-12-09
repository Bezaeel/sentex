# ONNX Sentiment Analysis with Rust - Complete Guide

## Overview
This guide walks you through exporting a DistilBERT sentiment analysis model to ONNX and using it in Rust for high-performance inference.

## Prerequisites

### Install uv (Modern Python Package Manager)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Part 1: Export Model to ONNX (Python)

### Method 1: Using the Export Script (Recommended)

```bash
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install transformers optimum[onnxruntime] onnx numpy

# Run the export script
uv run export_model_to_onnx.py
```

This will:
- Download DistilBERT model fine-tuned on SST-2 for sentiment analysis
- Export it to ONNX format
- Save to `sentiment_model_onnx/` directory
- Test the model to verify it works

### Method 2: Manual Steps

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Download and export model
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True  # Automatically converts to ONNX
)
model.save_pretrained("sentiment_model_onnx")

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer.save_pretrained("sentiment_model_onnx")
```

### Alternative: Using transformers.onnx CLI

```bash
python -m transformers.onnx \
    --model=distilbert-base-uncased-finetuned-sst-2-english \
    --feature=sequence-classification \
    sentiment_model_onnx/
```

## Part 2: Understanding the ONNX Model

### Files Generated

After export, you'll have:

```
sentiment_model_onnx/
├── model.onnx              # The ONNX model file (~250MB)
├── config.json             # Model configuration
├── tokenizer.json          # Tokenizer (fast tokenizer format)
├── tokenizer_config.json   # Tokenizer settings
├── vocab.txt               # Vocabulary file
└── special_tokens_map.json # Special tokens mapping
```

### Model Architecture

**DistilBERT** is a distilled (smaller) version of BERT:
- 40% smaller than BERT-base
- 60% faster inference
- Retains 95% of BERT's performance
- 6 transformer layers (vs 12 in BERT)
- 66M parameters

**Fine-tuned on SST-2** (Stanford Sentiment Treebank):
- Binary classification: Positive (1) vs Negative (0)
- Trained on 67,349 movie reviews
- ~91% accuracy on test set

### Model Inputs

The ONNX model expects:

1. **input_ids** (int64): Token IDs from tokenizer
   - Shape: `[batch_size, sequence_length]`
   - Max sequence length: 512 tokens
   
2. **attention_mask** (int64): Indicates real vs padded tokens
   - Shape: `[batch_size, sequence_length]`
   - Values: 1 for real tokens, 0 for padding

### Model Outputs

Returns **logits** (float32):
- Shape: `[batch_size, 2]`
- Index 0: Negative sentiment score
- Index 1: Positive sentiment score

To get probabilities, apply softmax:
```python
import numpy as np
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
```

## Part 3: Using ONNX Model in Rust

### Project Structure

```
rust-sentiment-api/
├── Cargo.toml
├── models/
│   ├── model.onnx
│   ├── tokenizer.json
│   └── config.json
└── src/
    ├── main.rs
    └── lib.rs (optional)
```

### Cargo.toml

```toml
[package]
name = "sentiment-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
ort = "2.0"                    # ONNX Runtime
tokenizers = "0.20"            # Hugging Face tokenizers
tokio = { version = "1", features = ["full"] }
axum = "0.7"                   # Web framework
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
```

### Basic Usage Example

```rust
use ort::{Session, Value, GraphOptimizationLevel};
use tokenizers::Tokenizer;
use ndarray::{Array, Axis};

fn main() -> anyhow::Result<()> {
    // Load ONNX model
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("models/model.onnx")?;
    
    // Load tokenizer
    let tokenizer = Tokenizer::from_file("models/tokenizer.json")?;
    
    // Tokenize input
    let text = "This movie is amazing!";
    let encoding = tokenizer.encode(text, true)?;
    
    // Prepare inputs
    let input_ids = Array::from_shape_vec(
        (1, encoding.get_ids().len()),
        encoding.get_ids().iter().map(|&x| x as i64).collect()
    )?;
    
    let attention_mask = Array::from_shape_vec(
        (1, encoding.get_attention_mask().len()),
        encoding.get_attention_mask().iter().map(|&x| x as i64).collect()
    )?;
    
    // Run inference
    let outputs = model.run(ort::inputs![
        "input_ids" => Value::from_array(input_ids)?,
        "attention_mask" => Value::from_array(attention_mask)?,
    ]?)?;
    
    // Process results
    let logits = outputs[0].try_extract_tensor::<f32>()?;
    let probs = softmax(logits.view());
    
    println!("Negative: {:.2}%, Positive: {:.2}%", 
        probs[0] * 100.0, probs[1] * 100.0);
    
    Ok(())
}

fn softmax(logits: ndarray::ArrayView1<f32>) -> Vec<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}
```

## Part 4: Building a REST API

See the complete implementation in `src/main.rs`.

Key endpoints:
- `GET /` - Health check
- `POST /predict` - Sentiment prediction

Example request:
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This restaurant is amazing!"}'
```

Example response:
```json
{
  "text": "This restaurant is amazing!",
  "label": "POSITIVE",
  "score": 0.9987
}
```

## Part 5: Testing with Yelp Dataset

```bash
# Run predictions on the Yelp dataset
cargo run --bin test-yelp
```

## Comparing with ML.NET

Key differences you'll notice:

### ML.NET
- Integrated training + inference in C#
- Model stored in .zip file
- Automatic feature engineering
- Great for .NET ecosystem

### Rust + ONNX
- Train in Python, inference in Rust
- Model in .onnx file (universal format)
- Manual preprocessing required
- Better performance and safety guarantees
- Language agnostic

### Performance Comparison

Typical results (1000 predictions):
- ML.NET: ~200-300ms
- Rust + ONNX (CPU): ~150-200ms
- Rust + ONNX (optimized): ~50-100ms

## Troubleshooting

### Model file too large
Use quantization to reduce model size:
```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("sentiment_model_onnx")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(save_dir="sentiment_model_quantized", quantization_config=qconfig)
```

### Slow inference
1. Enable CPU optimizations in ONNX Runtime
2. Use batching for multiple predictions
3. Consider using ONNX Runtime with DirectML/CUDA for GPU
4. Reduce max_sequence_length if possible

### Different results from Python
Check:
- Tokenizer settings match
- Padding/truncation settings
- Input preprocessing steps
- Softmax application

## Next Steps

1. **Day 2**: Build complete REST API with error handling
2. **Day 3**: Add batching and caching
3. **Day 4**: Performance optimization and benchmarking
4. **Week 2**: Advanced topics (quantization, GPU, ranking models)

## Resources

- [ONNX Runtime Rust Docs](https://docs.rs/ort/)
- [Hugging Face Optimum](https://huggingface.co/docs/optimum/)
- [Tokenizers Rust](https://docs.rs/tokenizers/)
- [Axum Web Framework](https://docs.rs/axum/)

## License

This is a learning project. The DistilBERT model is from Hugging Face and licensed under Apache 2.0.
