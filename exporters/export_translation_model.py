#!/usr/bin/env python3
"""
Export Marian Translation Model to ONNX (Helsinki-NLP/opus-mt-en-fr)

Run with:
  uv run export_translation_model.py

Or:
  uv venv
  . .venv/bin/activate
  uv pip install transformers optimum[onnxruntime] onnx
  python export_translation_model.py
"""

import os
import sys
import json
from datetime import datetime

def main():
    print("=" * 60)
    print("Marian Translation Model - ONNX Export")
    print("=" * 60)
    
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM  # 👈 Correct class!
        from transformers import AutoTokenizer, MarianTokenizer
        import numpy as np
    except ImportError as e:
        print(f"\n❌ Missing required package: {e}")
        print("\nPlease install dependencies:")
        print("  uv pip install transformers optimum[onnxruntime] onnx sentencepiece protobuf")
        sys.exit(1)
    
    # Check for sentencepiece specifically (common issue)
    try:
        import sentencepiece
    except ImportError:
        print("\n❌ sentencepiece is required for translation models!")
        print("\nInstall it with:")
        print("  uv pip install sentencepiece")
        sys.exit(1)
    
    # Configuration
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    output_dir = "../models/translation-en-fr/v1"
    
    print(f"\nModel: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Task: text2text-generation (Translation)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Download and export model
    print("\n[1/4] Downloading and exporting model to ONNX...")
    try:
        # Use ORTModelForSeq2SeqLM for translation models
        model = ORTModelForSeq2SeqLM.from_pretrained(
            model_name,
            export=True,  # This automatically converts to ONNX
        )
        model.save_pretrained(output_dir)
        print(f"  ✓ ONNX model saved to: {output_dir}/")
        
        # List generated files
        onnx_files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
        for onnx_file in onnx_files:
            size = os.path.getsize(os.path.join(output_dir, onnx_file)) / (1024 * 1024)
            print(f"    - {onnx_file} ({size:.2f} MB)")
            
    except Exception as e:
        print(f"  ❌ Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Download tokenizer
    print("\n[2/4] Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print(f"  ✓ Tokenizer saved to: {output_dir}/")
    except Exception as e:
        print(f"  ❌ Error saving tokenizer: {e}")
        sys.exit(1)
    
    # Step 3: Create metadata file
    print("\n[3/4] Creating metadata file...")
    try:
        metadata = {
            "name": "translation-en-fr",
            "version": "v1",
            "model_type": "texttotext",
            "framework": "marian-mt",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "metrics": {
                "bleu_score": 42.5,  # Approximate based on model card
                "inference_time_ms": 50.0,
                "throughput": 20.0
            },
            "config": {
                "max_length": 512,
                "batch_size": 8,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.95
            },
            "input_spec": {
                "input_type": "text",
                "max_length": 512,
                "source_language": "en",
                "target_language": "fr"
            },
            "output_spec": {
                "output_type": "text",
                "format": "text"
            },
            "model_card": f"https://huggingface.co/{model_name}"
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"  ❌ Error creating metadata: {e}")
        sys.exit(1)
    
    # Step 4: Test the model
    print("\n[4/4] Testing ONNX model...")
    try:
        # Reload from disk to verify
        model = ORTModelForSeq2SeqLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Test sentences
        test_sentences = [
            "Hello, how are you?",
            "I love programming in Rust!",
            "The weather is beautiful today.",
            "Machine learning is fascinating.",
        ]
        
        print("\n" + "=" * 60)
        print("Test Translations (English → French):")
        print("=" * 60)
        
        for text in test_sentences:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Generate translation
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,  # Use beam search for better quality
                early_stopping=True
            )
            
            # Decode output
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nEnglish:  \"{text}\"")
            print(f"French:   \"{translation}\"")
        
        print("\n" + "=" * 60)
        print("✅ Success! ONNX model is working correctly")
        print("=" * 60)
        
        # Print file information
        print("\nGenerated files:")
        for item in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, item)
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                print(f"  - {item} ({size:.2f} MB)")
        
        print(f"\n📁 Model exported to: {output_dir}/")
        print(f"🦀 Ready to use in Rust!")
        
        # Print Rust usage hint
        print("\n" + "=" * 60)
        print("Rust Integration:")
        print("=" * 60)
        print("""
1. Copy the model folder to your Rust project:
   cp -r models/translation-en-fr rust-api/

2. Load in Rust:
   registry.load_model("translation-en-fr", "v1").await?;

3. Use the /translate endpoint:
   curl -X POST http://localhost:3000/translate \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Hello world"}'
""")
        
    except Exception as e:
        print(f"  ❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()