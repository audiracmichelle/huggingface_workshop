"""
Exploring Opensource LLMs on Huggingface
==========================================

This script demonstrates how to explore and use opensource Large Language Models (LLMs)
from Huggingface Hub, including loading model weights and performing inference.

Usage:
    python explore_models.py

Requirements:
    - transformers
    - torch
    - huggingface-hub
"""

import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, list_models, model_info
import torch
import warnings
warnings.filterwarnings('ignore')


def check_environment():
    """Check the environment setup."""
    print("=" * 80)
    print("Environment Check")
    print("=" * 80)
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()


def explore_models():
    """Explore available models on Huggingface Hub."""
    print("=" * 80)
    print("Exploring Top Text Generation Models")
    print("=" * 80)
    
    # Initialize the Huggingface Hub API
    api = HfApi()
    
    # Search for text-generation models
    models = list(list_models(
        task="text-generation",
        sort="downloads",
        limit=10
    ))
    
    print("\nTop 10 Most Downloaded Text Generation Models:")
    print("-" * 80)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.id}")
        print(f"   Downloads: {model.downloads if hasattr(model, 'downloads') else 'N/A'}")
        print(f"   Likes: {model.likes if hasattr(model, 'likes') else 'N/A'}")
    print()


def get_model_info(model_id="gpt2"):
    """Get detailed information about a specific model."""
    print("=" * 80)
    print(f"Model Information: {model_id}")
    print("=" * 80)
    
    info = model_info(model_id)
    
    print(f"Model ID: {info.id}")
    print(f"Task: {info.pipeline_tag}")
    print(f"Library: {info.library_name}")
    print(f"Downloads: {info.downloads}")
    print(f"Likes: {info.likes}")
    print(f"Tags: {info.tags[:5] if info.tags else 'N/A'}")
    print()


def load_model_weights(model_name="gpt2"):
    """
    Load model weights from Huggingface Hub.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (tokenizer, model)
    """
    print("=" * 80)
    print(f"Loading Model: {model_name}")
    print("=" * 80)
    print("This may take a moment as weights are downloaded...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✓ Model loaded")
    
    # Check model size
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_parameters:,}")
    print(f"  Model size (fp32): ~{num_parameters * 4 / 1e9:.2f} GB")
    print()
    
    return tokenizer, model


def generate_text(model, tokenizer, prompt="Artificial intelligence is"):
    """
    Generate text using the loaded model.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Text prompt for generation
    """
    print("=" * 80)
    print("Text Generation")
    print("=" * 80)
    print(f"Prompt: '{prompt}'")
    print("\nGenerated text:")
    print("-" * 80)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    print()


def compare_model_sizes():
    """Compare sizes of different models."""
    print("=" * 80)
    print("Model Size Comparison")
    print("=" * 80)
    
    models_to_compare = {
        "distilgpt2": "DistilGPT-2 (Distilled)",
        "gpt2": "GPT-2 (Base)",
    }
    
    for model_name, description in models_to_compare.items():
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"{description}:")
            print(f"  Model: {model_name}")
            print(f"  Parameters: {num_params:,}")
            print(f"  Size (fp32): ~{num_params * 4 / 1e9:.2f} GB")
            print()
            del model  # Free memory
        except Exception as e:
            print(f"{description}: Could not load - {e}")
            print()


def inspect_model_architecture(model, max_layers=5):
    """
    Inspect the architecture and weights of a model.
    
    Args:
        model: Loaded model
        max_layers: Maximum number of layers to display
    """
    print("=" * 80)
    print("Model Architecture (First Few Layers)")
    print("=" * 80)
    
    total_params = 0
    layer_count = 0
    
    for name, param in model.named_parameters():
        if layer_count >= max_layers:
            print(f"... ({len(list(model.named_parameters())) - max_layers} more layers)")
            break
            
        print(f"Layer: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Parameters: {param.numel():,}")
        print(f"  Dtype: {param.dtype}")
        print()
        
        total_params += param.numel()
        layer_count += 1
    
    print(f"Total parameters in model: {sum(p.numel() for p in model.parameters()):,}")
    print()


def demonstrate_pipeline_api():
    """Demonstrate the easier pipeline API."""
    from transformers import pipeline
    
    print("=" * 80)
    print("Using Pipeline API (Simplified Interface)")
    print("=" * 80)
    
    # Create a text generation pipeline
    generator = pipeline('text-generation', model='gpt2')
    
    prompts = [
        "The future of technology is",
        "Open source software enables",
        "Machine learning can"
    ]
    
    for prompt in prompts:
        result = generator(prompt, max_length=40, num_return_sequences=1, pad_token_id=50256)
        print(f"Prompt: {prompt}")
        print(f"Output: {result[0]['generated_text']}")
        print()


def main():
    """Main function to run all demonstrations."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "EXPLORING OPENSOURCE LLMs ON HUGGINGFACE" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # 1. Check environment
    check_environment()
    
    # 2. Explore available models
    explore_models()
    
    # 3. Get detailed model information
    get_model_info("gpt2")
    
    # 4. Load model weights
    tokenizer, model = load_model_weights("gpt2")
    
    # 5. Generate text
    generate_text(model, tokenizer, "Artificial intelligence is")
    generate_text(model, tokenizer, "The future of machine learning")
    
    # 6. Inspect model architecture
    inspect_model_architecture(model, max_layers=3)
    
    # 7. Compare model sizes
    compare_model_sizes()
    
    # 8. Demonstrate pipeline API
    demonstrate_pipeline_api()
    
    print("=" * 80)
    print("Exploration Complete!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Try different models from Huggingface Hub")
    print("  2. Experiment with different generation parameters")
    print("  3. Fine-tune models on your own data")
    print("  4. Explore model quantization for efficiency")
    print("\nResources:")
    print("  - Huggingface Model Hub: https://huggingface.co/models")
    print("  - Documentation: https://huggingface.co/docs")
    print("  - Transformers Library: https://github.com/huggingface/transformers")
    print()


if __name__ == "__main__":
    main()
