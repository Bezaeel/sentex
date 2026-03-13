#!/usr/bin/env python3
"""
Export DistilBERT Sentiment Model to ONNX
Run this on your local machine with: uv run export_tc_model.py

Or with uv pip:
  uv venv
  . .venv/bin/activate  # On Windows: .venv\Scripts\activate
  uv pip install transformers optimum[onnxruntime] onnx
  python export_tc_model.py
"""

import os
import sys

def main():
    print("=" * 60)
    print("DistilBERT Sentiment Analysis - ONNX Export")
    print("=" * 60)
    
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
        import numpy as np
    except ImportError as e:
        print(f"\n❌ Missing required package: {e}")
        print("\nPlease install dependencies:")
        print("  uv pip install transformers optimum[onnxruntime] onnx")
        sys.exit(1)
    
    # Configuration
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    output_dir = "sentiment_model_onnx"
    
    print(f"\nModel: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download and export model
    print("\n[1/3] Downloading and exporting model to ONNX...")
    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,  # This automatically converts to ONNX
        )
        model.save_pretrained(output_dir)
        print(f"  ✓ ONNX model saved to: {output_dir}/model.onnx")
    except Exception as e:
        print(f"  ❌ Error exporting model: {e}")
        sys.exit(1)
    
    # Step 2: Download tokenizer
    print("\n[2/3] Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print(f"  ✓ Tokenizer saved to: {output_dir}/")
    except Exception as e:
        print(f"  ❌ Error saving tokenizer: {e}")
        sys.exit(1)
    
    # Step 3: Test the model
    print("\n[3/3] Testing ONNX model...")
    try:
        # Reload from disk to verify
        model = ORTModelForSequenceClassification.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Test sentences
        test_sentences = [
            "This restaurant is amazing! Best food I've ever had!",
            "Terrible experience. The service was awful and food was cold.",
            "It was okay. Nothing special but not bad either.",
        ]
        
        print("\n" + "=" * 60)
        print("Test Predictions:")
        print("=" * 60)
        
        for text in test_sentences:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Run inference
            outputs = model(**inputs)
            logits = outputs.logits.detach().numpy()[0]
            
            # Calculate probabilities
            probs = np.exp(logits) / np.exp(logits).sum()
            
            # Get prediction
            predicted_idx = logits.argmax()
            labels = ["NEGATIVE", "POSITIVE"]
            
            print(f"\nText: \"{text}\"")
            print(f"  Prediction: {labels[predicted_idx]}")
            print(f"  Confidence: {probs[predicted_idx]:.1%}")
            print(f"  Scores: Negative={probs[0]:.1%}, Positive={probs[1]:.1%}")
        
        print("\n" + "=" * 60)
        print("✅ Success! ONNX model is working correctly")
        print("=" * 60)
        
        # Print file information
        print("\nGenerated files:")
        for item in os.listdir(output_dir):
            path = os.path.join(output_dir, item)
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                print(f"  - {item} ({size:.2f} MB)")
        
        print(f"\nModel is ready to use in Rust!")
        print(f"Copy the '{output_dir}' folder to your Rust project.")
        
    except Exception as e:
        print(f"  ❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
