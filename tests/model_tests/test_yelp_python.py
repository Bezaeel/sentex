#!/usr/bin/env python3
"""
Test the ONNX sentiment model on Yelp dataset
Compare with Rust implementation
"""
import time
import numpy as np
from pathlib import Path

def load_yelp_dataset(filepath="yelp_labelled.txt"):
    """Load the Yelp dataset"""
    reviews = []
    labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text, label = parts
                if text.strip():
                    reviews.append(text.strip())
                    labels.append(int(label))

    return reviews, labels

def test_model():
    """Test the ONNX model on Yelp dataset"""
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ Missing required packages!")
        print("Install with: uv pip install transformers optimum[onnxruntime]")
        return

    print("=" * 50)
    print("🧪 Testing ONNX Model on Yelp Dataset (Python)")
    print("=" * 50)
    print()

    # Load dataset
    print("📂 Loading Yelp dataset...")
    reviews, labels = load_yelp_dataset()
    print(f"   Loaded {len(reviews)} reviews\n")

    # Load model
    print("📁 Loading ONNX model...")
    model_path = "../../models"

    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run: python export_model_to_onnx.py")
        return

    model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("   ✓ Model loaded\n")

    # Run predictions
    print(f"🔮 Running predictions on {len(reviews)} reviews...")
    start_time = time.time()

    predictions = []
    confidences = []

    for i, text in enumerate(reviews):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i + 1}/{len(reviews)}")

        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits.detach().numpy()[0]

        # Calculate probabilities
        probs = np.exp(logits) / np.exp(logits).sum()
        pred = logits.argmax()

        predictions.append(pred)
        confidences.append(probs[pred])

    elapsed = time.time() - start_time
    print(f"   ✓ Completed in {elapsed:.2f}s")
    print(f"   Average: {elapsed * 1000 / len(reviews):.2f}ms per prediction\n")

    # Calculate metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    confidences = np.array(confidences)

    correct = (predictions == labels).sum()
    accuracy = correct / len(labels)

    # Confusion matrix
    tp = ((predictions == 1) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("=" * 50)
    print("📊 Results")
    print("=" * 50)
    print()

    print("Overall Performance:")
    print(f"  Total Reviews:    {len(labels)}")
    print(f"  Correct:          {correct}")
    print(f"  Incorrect:        {len(labels) - correct}")
    print(f"  Accuracy:         {accuracy * 100:.2f}%\n")

    print("Confusion Matrix:")
    print("                 Predicted Positive  Predicted Negative")
    print(f"  Actual Positive      {tp:4}                {fn:4}")
    print(f"  Actual Negative      {fp:4}                {tn:4}\n")

    print("Performance Metrics:")
    print(f"  Precision:        {precision * 100:.2f}%")
    print(f"  Recall:           {recall * 100:.2f}%")
    print(f"  F1 Score:         {f1 * 100:.2f}%\n")

    # Sample predictions
    print("=" * 50)
    print("📝 Sample Predictions")
    print("=" * 50)
    print()

    print("✓ Correct Predictions (5 examples):")
    correct_indices = np.where(predictions == labels)[0][:5]
    for idx in correct_indices:
        label_str = "POSITIVE" if labels[idx] == 1 else "NEGATIVE"
        text_preview = reviews[idx][:60]
        print(f'  Text: "{text_preview}"')
        print(f"  Label: {label_str} | Confidence: {confidences[idx] * 100:.1f}%\n")

    print("✗ Incorrect Predictions (5 examples):")
    incorrect_indices = np.where(predictions != labels)[0][:5]
    for idx in incorrect_indices:
        actual = "POSITIVE" if labels[idx] == 1 else "NEGATIVE"
        predicted = "POSITIVE" if predictions[idx] == 1 else "NEGATIVE"
        text_preview = reviews[idx][:60]
        print(f'  Text: "{text_preview}"')
        print(f"  Actual: {actual} | Predicted: {predicted} | Confidence: {confidences[idx] * 100:.1f}%\n")

    # Low confidence
    low_conf_indices = np.where(confidences < 0.7)[0]
    if len(low_conf_indices) > 0:
        print(f"⚠ Low Confidence Predictions ({len(low_conf_indices)} total):")
        for idx in low_conf_indices[:5]:
            predicted = "POSITIVE" if predictions[idx] == 1 else "NEGATIVE"
            text_preview = reviews[idx][:60]
            correct_mark = "✓" if predictions[idx] == labels[idx] else "✗"
            print(f'  Text: "{text_preview}"')
            print(f"  Predicted: {predicted} | Confidence: {confidences[idx] * 100:.1f}% {correct_mark}\n")

    print("=" * 50)
    print("✅ Testing Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_model()
