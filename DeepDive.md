# Deep Dive: Rust ONNX Sentiment Analysis Code

This guide provides a line-by-line explanation of how to load and run an ONNX model in Rust.

---

## Complete Code Overview

```rust
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // 1. Load ONNX model
    // 2. Load tokenizer
    // 3. Tokenize input
    // 4. Prepare tensors
    // 5. Run inference
    // 6. Process results
}
```

---

## Part 1: Imports (Use Statements)

### 1.1 ndarray::Array2

```rust
use ndarray::Array2;
```

**What it is:**
- `ndarray` is Rust's equivalent of NumPy
- `Array2<T>` is a 2-dimensional array (matrix)
- Generic type `T` can be `i64`, `f32`, etc.

**Why we need it:**
- ONNX models expect inputs as multi-dimensional arrays (tensors)
- `Array2<i64>` represents a 2D array of 64-bit integers
- Shape is `[batch_size, sequence_length]`

**Example:**
```rust
// Create a 2x3 array (2 rows, 3 columns)
let arr: Array2<i64> = Array2::from_shape_vec(
    (2, 3),              // Shape: 2 rows, 3 cols
    vec![1, 2, 3, 4, 5, 6]  // Data in row-major order
)?;

// Result:
// [[1, 2, 3],
//  [4, 5, 6]]
```

### 1.2 ort (ONNX Runtime)

```rust
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
```

**What they are:**

- **`Session`**: Represents a loaded ONNX model
  - Think of it as the "runtime" for your model
  - Handles inference (predictions)
  
- **`GraphOptimizationLevel`**: Optimization settings
  - `Level1`: Basic optimizations
  - `Level2`: Extended optimizations
  - `Level3`: All optimizations (highest performance)
  
- **`Tensor`**: Wrapper around arrays for ONNX
  - Converts Rust arrays to ONNX-compatible format
  - Handles data layout and type conversion

**Why we need them:**
- `Session`: To load and run the ONNX model
- `GraphOptimizationLevel`: To optimize inference speed
- `Tensor`: To pass data to the model in the correct format

### 1.3 tokenizers

```rust
use tokenizers::Tokenizer;
```

**What it is:**
- Rust bindings to Hugging Face tokenizers
- Converts text → numbers (token IDs)
- Same tokenizers used in Python transformers library

**Why we need it:**
- ML models can't process raw text
- Text must be converted to numbers first
- The tokenizer knows the model's vocabulary

**Example:**
```rust
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let text = "Hello world";
let encoding = tokenizer.encode(text, true)?;

// encoding.get_ids() = [101, 7592, 2088, 102]
// 101 = [CLS], 7592 = "hello", 2088 = "world", 102 = [SEP]
```

---

## Part 2: Loading the Model

```rust
let mut model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file("models/model.onnx")?;
```

### Line-by-Line Breakdown:

#### 2.1 `Session::builder()?`

```rust
Session::builder()?
```

**What it does:**
- Creates a `SessionBuilder` - a builder pattern for configuring sessions
- The `?` operator propagates errors if initialization fails

**Why builder pattern:**
- Allows chaining configuration options
- Makes it clear what settings are being applied
- Flexible - add/remove options as needed

**Equivalent to:**
```rust
let builder = Session::builder();
match builder {
    Ok(b) => b,
    Err(e) => return Err(e),
}
```

#### 2.2 `.with_optimization_level(GraphOptimizationLevel::Level3)?`

```rust
.with_optimization_level(GraphOptimizationLevel::Level3)?
```

**What it does:**
- Enables aggressive optimizations for the computation graph
- Level3 = maximum performance optimizations

**Optimizations include:**
- Constant folding (compute constants at load time)
- Operator fusion (combine multiple ops into one)
- Memory layout optimization
- Dead code elimination

**Performance impact:**
```rust
Level1: ~10-15ms per prediction
Level2: ~8-12ms per prediction  
Level3: ~5-10ms per prediction  (Fastest!)
```

**Trade-off:**
- Level3 takes longer to load (1-2 extra seconds)
- But much faster inference
- Worth it for production!

#### 2.3 `.commit_from_file("models/model.onnx")?`

```rust
.commit_from_file("models/model.onnx")?
```

**What it does:**
- Loads the ONNX model file from disk
- Parses the model structure
- Initializes weights and operators
- Returns the configured `Session`

**Behind the scenes:**
1. Opens and reads the .onnx file (binary format)
2. Parses the protobuf structure
3. Loads all weights into memory
4. Applies optimizations (Level3)
5. Prepares for inference

**File size:**
- DistilBERT model: ~255MB
- Contains: weights, biases, layer configurations
- Format: ONNX protobuf

#### 2.4 `let mut model`

```rust
let mut model = ...
```

**Why `mut`?**
- `Session` might need internal state updates during inference
- Some ONNX operations maintain state between runs
- Required by the `run()` method signature

**Type:**
```rust
model: Session  // Fully loaded ONNX model ready for inference
```

---

## Part 3: Loading the Tokenizer

```rust
let tokenizer = Tokenizer::from_file("models/tokenizer.json")
    .map_err(|e| anyhow::anyhow!("{}", e))?;
```

### 3.1 `Tokenizer::from_file("models/tokenizer.json")`

**What it does:**
- Loads the tokenizer configuration from JSON
- Contains vocabulary, special tokens, tokenization rules

**tokenizer.json structure:**
```json
{
  "version": "1.0",
  "truncation": {"max_length": 512},
  "padding": {"strategy": "BatchLongest"},
  "added_tokens": [
    {"id": 0, "content": "[PAD]"},
    {"id": 101, "content": "[CLS]"},
    {"id": 102, "content": "[SEP]"}
  ],
  "vocab": {
    "[PAD]": 0,
    "the": 1996,
    "movie": 3185,
    "amazing": 2307,
    ...
  }
}
```

**Vocabulary size:**
- DistilBERT: 30,522 tokens
- Includes: words, subwords, special tokens

### 3.2 `.map_err(|e| anyhow::anyhow!("{}", e))?`

**What it does:**
- Error conversion - transforms tokenizer errors into anyhow errors
- Makes error handling consistent across the codebase

**Breaking it down:**

```rust
.map_err(|e| anyhow::anyhow!("{}", e))
```

- `map_err`: Transform the error type
- `|e|`: Closure that takes the error
- `anyhow::anyhow!("{}", e)`: Create anyhow error from message

**Why needed:**
- `Tokenizer::from_file()` returns its own error type
- `main()` returns `anyhow::Result<()>`
- Need to convert between error types

**Equivalent to:**
```rust
let tokenizer = match Tokenizer::from_file("models/tokenizer.json") {
    Ok(t) => t,
    Err(e) => return Err(anyhow::anyhow!("{}", e)),
};
```

---

## Part 4: Tokenizing Input

```rust
let text = "This movie is amazing!";
let encoding = tokenizer
    .encode(text, true)
    .map_err(|e| anyhow::anyhow!("{}", e))?;
```

### 4.1 The Input Text

```rust
let text = "This movie is amazing!";
```

**Type:**
```rust
text: &str  // String slice (borrowed string)
```

### 4.2 `tokenizer.encode(text, true)`

```rust
let encoding = tokenizer.encode(text, true)
```

**Parameters:**
- `text`: The string to tokenize
- `true`: Add special tokens ([CLS], [SEP])

**What happens:**

1. **Text normalization:**
   ```
   "This movie is amazing!" → "this movie is amazing!"
   ```

2. **Tokenization (split into words/subwords):**
   ```
   ["this", "movie", "is", "amazing", "!"]
   ```

3. **Convert to IDs:**
   ```
   [101, 2023, 3185, 2003, 2307, 999, 102]
   ```
   - 101 = [CLS] (start token)
   - 2023 = "this"
   - 3185 = "movie"
   - 2003 = "is"
   - 2307 = "amazing"
   - 999 = "!"
   - 102 = [SEP] (end token)

4. **Create attention mask:**
   ```
   [1, 1, 1, 1, 1, 1, 1]
   ```
   All 1s = all tokens are real (no padding)

**Return value:**
```rust
encoding: Encoding  // Contains: ids, attention_mask, tokens, etc.
```

### 4.3 What's in the Encoding?

```rust
// Available methods:
encoding.get_ids()              // [101, 2023, 3185, ...]
encoding.get_attention_mask()   // [1, 1, 1, ...]
encoding.get_tokens()           // ["[CLS]", "this", "movie", ...]
encoding.get_type_ids()         // [0, 0, 0, ...] (segment IDs)
encoding.get_offsets()          // Character positions in original text
```

---

## Part 5: Preparing Tensors

```rust
let seq_len = encoding.get_ids().len();

let input_ids: Array2<i64> = Array2::from_shape_vec(
    (1, seq_len),
    encoding.get_ids().iter().map(|&x| x as i64).collect(),
)?;

let attention_mask: Array2<i64> = Array2::from_shape_vec(
    (1, seq_len),
    encoding.get_attention_mask().iter().map(|&x| x as i64).collect(),
)?;
```

### 5.1 Get Sequence Length

```rust
let seq_len = encoding.get_ids().len();
```

**Example:**
```rust
// For "This movie is amazing!"
seq_len = 7  // [CLS] this movie is amazing ! [SEP]
```

### 5.2 Create input_ids Tensor

```rust
let input_ids: Array2<i64> = Array2::from_shape_vec(
    (1, seq_len),
    encoding.get_ids().iter().map(|&x| x as i64).collect(),
)?;
```

**Breaking it down:**

#### Step 1: Get token IDs
```rust
encoding.get_ids()
// Returns: &[u32] = [101, 2023, 3185, 2003, 2307, 999, 102]
```

#### Step 2: Convert u32 → i64
```rust
.iter().map(|&x| x as i64).collect()
```

- `.iter()`: Create iterator over the slice
- `|&x|`: Closure parameter - pattern match to get value
- `x as i64`: Cast from u32 to i64 (ONNX expects i64)
- `.collect()`: Collect into Vec<i64>

**Result:**
```rust
vec![101i64, 2023i64, 3185i64, 2003i64, 2307i64, 999i64, 102i64]
```

#### Step 3: Reshape to 2D array
```rust
Array2::from_shape_vec((1, seq_len), ...)
```

**Shape `(1, seq_len)`:**
- First dimension = batch size (1 = single sentence)
- Second dimension = sequence length (7 tokens)

**Visualization:**
```
Before reshape: [101, 2023, 3185, 2003, 2307, 999, 102]

After reshape (1, 7):
[[101, 2023, 3185, 2003, 2307, 999, 102]]
```

**Why reshape?**
- Models expect batch dimension even for single input
- Allows processing multiple sentences at once
- Standard ML convention: `[batch_size, features]`

### 5.3 Create attention_mask Tensor

```rust
let attention_mask: Array2<i64> = Array2::from_shape_vec(
    (1, seq_len),
    encoding.get_attention_mask().iter().map(|&x| x as i64).collect(),
)?;
```

**Same process as input_ids but with attention mask:**

```rust
// From: [1, 1, 1, 1, 1, 1, 1] (all tokens are real)
// To:   [[1, 1, 1, 1, 1, 1, 1]] (2D array)
```

**What is attention mask?**

Tells the model which tokens to pay attention to:
- `1` = real token, pay attention
- `0` = padding token, ignore

**Example with padding:**
```rust
// Input: "Hello world"
// Tokens: [CLS] hello world [SEP] [PAD] [PAD] [PAD] [PAD]
// Mask:   [  1,   1,    1,   1,    0,    0,    0,    0  ]
//         ↑ real tokens ↑    ↑ ignore padding ↑
```

### 5.4 Why i64 and not f32?

```rust
Array2<i64>  // Not Array2<f32>
```

**Reason:**
- Token IDs are integers (101, 2023, 3185, ...)
- They're indices into the embedding table
- Floats would be wasteful and incorrect

**Model internals:**
```rust
// Input layer does this:
embedding_table[token_id] → embedding_vector

// Example:
embedding_table[2023] → [0.123, -0.456, 0.789, ...]  // 768 dimensions
```

---

## Part 6: Running Inference

```rust
let input_ids_tensor = Tensor::from_array(input_ids)?;
let attention_mask_tensor = Tensor::from_array(attention_mask)?;

let outputs = model.run(ort::inputs![
    "input_ids" => input_ids_tensor,
    "attention_mask" => attention_mask_tensor,
])?;
```

### 6.1 Convert Arrays to ONNX Tensors

```rust
let input_ids_tensor = Tensor::from_array(input_ids)?;
let attention_mask_tensor = Tensor::from_array(attention_mask)?;
```

**What this does:**
- Wraps ndarray in ONNX-compatible format
- Handles memory layout conversion
- Prepares for passing to ONNX Runtime

**Type transformation:**
```rust
Array2<i64>  →  Tensor<'_, i64>
```

### 6.2 The `ort::inputs![]` Macro

```rust
ort::inputs![
    "input_ids" => input_ids_tensor,
    "attention_mask" => attention_mask_tensor,
]
```

**What it does:**
- Creates a map of input names → tensors
- Input names must match model's expected names
- Order doesn't matter (it's a map)

**Equivalent to:**
```rust
let mut inputs = HashMap::new();
inputs.insert("input_ids", input_ids_tensor);
inputs.insert("attention_mask", attention_mask_tensor);
```

**Model's input specification:**
```
Input 0: "input_ids" - Int64[batch_size, sequence_length]
Input 1: "attention_mask" - Int64[batch_size, sequence_length]
```

### 6.3 `model.run(...)?`

```rust
let outputs = model.run(...)?;
```

**What happens:**

1. **Input validation:**
   - Check input names match
   - Check shapes are compatible
   - Check data types are correct

2. **Forward pass through model:**
   ```
   Input → Embeddings → Transformer Layers → Output
   ```

3. **Model computation (DistilBERT):**
   - 6 transformer layers
   - Self-attention mechanisms
   - Feed-forward networks
   - Layer normalization

4. **Returns output tensors:**
   ```rust
   outputs: Vec<OrtOutput>  // Array of output tensors
   ```

**Performance:**
- On CPU: ~10-15ms
- On GPU: ~2-5ms
- Most time spent in transformer layers

---

## Part 7: Processing Results

```rust
let logits = outputs[0].try_extract_tensor::<f32>()?;
let (_, logits_data) = logits;
let probs = softmax(logits_data);

println!(
    "Negative: {:.2}%, Positive: {:.2}%",
    probs[0] * 100.0,
    probs[1] * 100.0
);
```

### 7.1 Extract Output Tensor

```rust
let logits = outputs[0].try_extract_tensor::<f32>()?;
```

**What it does:**
- Gets first output (index 0)
- Extracts as f32 tensor
- `try_` prefix means it can fail if wrong type

**Output structure:**
```rust
outputs[0] = logits  // Raw model scores
// No outputs[1], model only has one output
```

### 7.2 Destructure the Tensor

```rust
let (_, logits_data) = logits;
```

**What `try_extract_tensor` returns:**
```rust
(TensorShape, &[f32])
```

- First element: Shape information `(1, 2)`
- Second element: Actual data `&[f32]`

**We only need the data:**
```rust
let (_, logits_data) = logits;
//    ^              ^
//    |              └── The data we want
//    └── Shape (discarded with _)
```

**logits_data contains:**
```rust
// For "This movie is amazing!"
logits_data = [-2.314, 2.891]
//             ↑ negative  ↑ positive
```

**What are logits?**
- Raw unnormalized scores from the model
- Not probabilities yet (can be negative, > 1)
- Higher value = stronger prediction

### 7.3 Apply Softmax

```rust
let probs = softmax(logits_data);
```

**Converts logits → probabilities:**
```rust
Input:  [-2.314, 2.891]
Output: [0.0013, 0.9987]  // Sums to 1.0
```

### 7.4 Print Results

```rust
println!(
    "Negative: {:.2}%, Positive: {:.2}%",
    probs[0] * 100.0,
    probs[1] * 100.0
);
```

**Output:**
```
Negative: 0.13%, Positive: 99.87%
```

**Format specifier `{:.2}`:**
- `.2` = 2 decimal places
- `%` = literal percent sign

---

## Part 8: The Softmax Function

```rust
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}
```

### 8.1 Function Signature

```rust
fn softmax(logits: &[f32]) -> Vec<f32>
```

**Parameters:**
- `logits: &[f32]` - Borrowed slice of floats (the raw scores)

**Returns:**
- `Vec<f32>` - Owned vector of probabilities

### 8.2 Find Maximum Value

```rust
let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
```

**What it does:**
- Finds the largest value in logits
- Used for numerical stability

**Breaking it down:**

```rust
logits.iter()  // Create iterator: [-2.314, 2.891]
.fold(
    f32::NEG_INFINITY,  // Initial accumulator value
    |a, &b| a.max(b)    // For each element, keep the max
)
```

**Execution trace:**
```rust
Step 1: a = NEG_INFINITY, b = -2.314 → max = -2.314
Step 2: a = -2.314,       b = 2.891  → max = 2.891
Result: max = 2.891
```

**Why find max?**

**Numerical stability!**

Without subtracting max:
```rust
// For large numbers:
logits = [100, 101]
exp(100) = 2.7 × 10^43  // Overflow! Returns infinity
```

With subtracting max:
```rust
logits = [100, 101]
max = 101
[100-101, 101-101] = [-1, 0]
exp(-1) = 0.368, exp(0) = 1.0  // Safe!
```

### 8.3 Exponentiate (Subtract Max for Stability)

```rust
let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
```

**Step by step:**

1. **Subtract max from each logit:**
   ```rust
   logits = [-2.314, 2.891]
   max = 2.891
   
   -2.314 - 2.891 = -5.205
   2.891 - 2.891 = 0.0
   
   Result: [-5.205, 0.0]
   ```

2. **Apply exponential function:**
   ```rust
   exp(-5.205) = 0.0055
   exp(0.0) = 1.0
   
   exp = [0.0055, 1.0]
   ```

**Why exponential?**
- Converts any real number to positive number
- Amplifies differences (larger gaps in output)
- Mathematical requirement for softmax

### 8.4 Sum All Exponentials

```rust
let sum: f32 = exp.iter().sum();
```

**Calculate:**
```rust
exp = [0.0055, 1.0]
sum = 0.0055 + 1.0 = 1.0055
```

### 8.5 Normalize (Divide by Sum)

```rust
exp.iter().map(|&x| x / sum).collect()
```

**Final step:**
```rust
exp = [0.0055, 1.0]
sum = 1.0055

[0.0055 / 1.0055, 1.0 / 1.0055]
= [0.0055, 0.9945]
```

**Result: Probabilities that sum to 1.0!**

### 8.6 Complete Softmax Example

```rust
// Input
logits = [-2.314, 2.891]

// Step 1: Find max
max = 2.891

// Step 2: Subtract max & exponentiate
x - max: [-5.205, 0.0]
exp:     [0.0055, 1.0]

// Step 3: Sum
sum = 1.0055

// Step 4: Normalize
probs = [0.0055, 0.9945]

// Verify: 0.0055 + 0.9945 = 1.0 ✓
```

---

## Complete Data Flow Visualization

```
"This movie is amazing!"
         ↓
    [Tokenizer]
         ↓
IDs:  [101, 2023, 3185, 2003, 2307, 999, 102]
Mask: [1, 1, 1, 1, 1, 1, 1]
         ↓
    [Reshape to 2D]
         ↓
input_ids:      [[101, 2023, 3185, 2003, 2307, 999, 102]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1]]
         ↓
    [Convert to ONNX Tensors]
         ↓
    [ONNX Model Forward Pass]
    6 Transformer Layers
         ↓
logits: [[-2.314, 2.891]]
         ↓
    [Softmax]
         ↓
probs: [[0.0013, 0.9987]]
         ↓
Output: Negative: 0.13%, Positive: 99.87%
```

---

## Memory Layout Deep Dive

### How Arrays are Stored in Memory

```rust
Array2::from_shape_vec((1, 7), vec![101, 2023, 3185, 2003, 2307, 999, 102])
```

**Memory representation (row-major order):**
```
Address | Value | Meaning
--------|-------|--------
0x1000  | 101   | Row 0, Col 0
0x1004  | 2023  | Row 0, Col 1
0x1008  | 3185  | Row 0, Col 2
0x100C  | 2003  | Row 0, Col 3
0x1010  | 2307  | Row 0, Col 4
0x1014  | 999   | Row 0, Col 5
0x1018  | 102   | Row 0, Col 6
```

**Shape (1, 7):**
- Stride: [7, 1]
  - Moving to next row: skip 7 elements
  - Moving to next column: skip 1 element

---

## Performance Characteristics

### Time Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| Load model | O(1) | One-time cost (~200ms) |
| Tokenization | O(n) | n = text length (~0.1ms per char) |
| Array creation | O(n) | n = sequence length (~0.01ms) |
| ONNX inference | O(n²) | Transformer attention (~10-15ms) |
| Softmax | O(k) | k = num classes (< 0.01ms) |

**Total per prediction: ~10-15ms**

### Memory Usage

```rust
// DistilBERT sentiment model
Model weights:      255 MB  (loaded once)
Input tensors:      ~1 KB   (per prediction)
Intermediate:       ~50 MB  (during inference)
Output:             ~8 bytes (2 floats)

Peak memory: ~305 MB
```

---

## Common Pitfalls and Solutions

### 1. Shape Mismatch

**Problem:**
```rust
// Wrong shape
Array2::from_shape_vec((7, 1), ids)  // ❌ Column vector

// Correct shape
Array2::from_shape_vec((1, 7), ids)  // ✓ Row vector
```

### 2. Type Mismatch

**Problem:**
```rust
// u32 from tokenizer
let ids: Vec<u32> = encoding.get_ids();

// But ONNX expects i64
let tensor: Array2<i64> = ...  // Must cast!
```

**Solution:**
```rust
.map(|&x| x as i64)  // Explicit cast
```

### 3. Forgetting Special Tokens

**Problem:**
```rust
tokenizer.encode(text, false)  // ❌ No [CLS]/[SEP]
```

**Solution:**
```rust
tokenizer.encode(text, true)  // ✓ Adds special tokens
```

### 4. Not Using Optimization

**Problem:**
```rust
Session::builder()?
    // No optimization level
    .commit_from_file("model.onnx")?
// Inference: ~30ms (slow!)
```

**Solution:**
```rust
Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file("model.onnx")?
// Inference: ~10ms (3x faster!)
```

---

## Comparing to Python

### Python (transformers)
```python
from transformers import pipeline

model = pipeline("sentiment-analysis")
result = model("This movie is amazing!")
# ~25ms per prediction
```

### Rust (ort)
```rust
let model = Session::builder()? ...
// ~10ms per prediction (2.5x faster!)
```

**Why Rust is faster:**
- No Python interpreter overhead
- Better memory layout
- Zero-cost abstractions
- No GC pauses

---

## Further Reading

**Rust Concepts:**
- [The Rust Book - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Error Handling with ?](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html)
- [Iterators](https://doc.rust-lang.org/book/ch13-02-iterators.html)

**ONNX Runtime:**
- [ort crate docs](https://docs.rs/ort/)
- [ONNX format spec](https://onnx.ai/onnx/intro/)
- [Optimization guide](https://onnxruntime.ai/docs/performance/)

**Transformers:**
- [DistilBERT paper](https://arxiv.org/abs/1910.01108)
- [Attention mechanism](https://arxiv.org/abs/1706.03762)
- [Tokenization explained](https://huggingface.co/docs/transformers/tokenizer_summary)

---

**Questions? Try modifying the code and see what happens!**
