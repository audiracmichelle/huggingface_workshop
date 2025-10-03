#!/usr/bin/env python3
"""
Simple Demo: Quick Text Generation with Huggingface Models
===========================================================

This is the simplest possible example to get started with Huggingface LLMs.
Run this script to see text generation in action!

Usage:
    python simple_demo.py
    
Or make it executable and run directly:
    chmod +x simple_demo.py
    ./simple_demo.py
"""

def main():
    print("\n" + "="*80)
    print("🤗 Huggingface LLM Quick Demo")
    print("="*80 + "\n")
    
    print("Loading model... (this may take a moment on first run)")
    print("The model will be cached for faster loading next time.\n")
    
    try:
        from transformers import pipeline
        import warnings
        warnings.filterwarnings('ignore')
        
        # Create a text generation pipeline with a small model
        generator = pipeline('text-generation', model='distilgpt2')
        
        print("✓ Model loaded successfully!\n")
        print("="*80)
        print("Generating text samples...")
        print("="*80 + "\n")
        
        # Example prompts
        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "Machine learning enables us to",
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Example {i}:")
            print(f"Prompt: '{prompt}'")
            print("-" * 80)
            
            # Generate text
            result = generator(
                prompt,
                max_length=40,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            print(f"Generated: {generated_text}")
            print()
        
        print("="*80)
        print("Demo completed successfully! 🎉")
        print("="*80 + "\n")
        
        print("What you just saw:")
        print("  • Loaded a lightweight LLM (DistilGPT-2)")
        print("  • Generated creative text continuations")
        print("  • Used the simple pipeline API\n")
        
        print("Next steps:")
        print("  1. Run 'python explore_models.py' for more examples")
        print("  2. Open 'explore_llms.ipynb' in Jupyter for interactive learning")
        print("  3. Try different models from https://huggingface.co/models")
        print("  4. Experiment with different prompts and parameters\n")
        
        print("Tips:")
        print("  • Adjust 'temperature' (0.1-1.0) to control randomness")
        print("  • Increase 'max_length' for longer outputs")
        print("  • Try 'num_return_sequences' > 1 for multiple variations\n")
        
    except ImportError as e:
        print("❌ Error: Required packages not installed")
        print("\nPlease install the required packages:")
        print("  pip install -r requirements.txt\n")
        print(f"Details: {e}\n")
        return 1
        
    except Exception as e:
        print(f"❌ An error occurred: {e}\n")
        print("Troubleshooting:")
        print("  • Make sure you have an internet connection")
        print("  • Check that you have enough disk space (2GB+)")
        print("  • Try running: pip install --upgrade transformers torch\n")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
