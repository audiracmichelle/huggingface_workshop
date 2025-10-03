"""
Practical Examples: Model Weights Usage Patterns
=================================================

This script demonstrates practical patterns for working with Huggingface model weights,
including efficient loading, saving, and sharing models.

Usage:
    python model_weights_usage.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path


class ModelWeightsManager:
    """Manager class for handling model weights efficiently."""
    
    def __init__(self, cache_dir="./model_cache"):
        """
        Initialize the model weights manager.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_model(self, model_name, precision="fp32"):
        """
        Download and cache a model with specified precision.
        
        Args:
            model_name: Name of the model on Huggingface Hub
            precision: Model precision - 'fp32', 'fp16', or 'int8'
            
        Returns:
            tuple: (tokenizer, model)
        """
        print(f"\n{'='*80}")
        print(f"Downloading Model: {model_name}")
        print(f"Precision: {precision}")
        print(f"{'='*80}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        print("✓ Tokenizer loaded")
        
        # Load model with specified precision
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
        }.get(precision, torch.float32)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded")
        
        # Model statistics
        num_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
        }.get(precision, 4)
        size_gb = (num_params * bytes_per_param) / 1e9
        
        print(f"\nModel Statistics:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Memory size: ~{size_gb:.2f} GB")
        
        return tokenizer, model
    
    def save_model_locally(self, model, tokenizer, save_path):
        """
        Save model and tokenizer to local directory.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            save_path: Directory path to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Saving Model to: {save_path}")
        print(f"{'='*80}")
        
        # Save model
        model.save_pretrained(save_path)
        print("✓ Model saved")
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        print("✓ Tokenizer saved")
        
        # List saved files
        files = list(save_path.glob("*"))
        print(f"\nSaved files:")
        for f in files:
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name} ({size_mb:.2f} MB)")
    
    def load_local_model(self, model_path):
        """
        Load model from local directory.
        
        Args:
            model_path: Path to the saved model directory
            
        Returns:
            tuple: (tokenizer, model)
        """
        model_path = Path(model_path)
        
        print(f"\n{'='*80}")
        print(f"Loading Model from: {model_path}")
        print(f"{'='*80}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("✓ Model loaded")
        
        return tokenizer, model
    
    def inspect_weights(self, model, num_layers=3):
        """
        Inspect model weight details.
        
        Args:
            model: Model to inspect
            num_layers: Number of layers to display
        """
        print(f"\n{'='*80}")
        print("Model Weight Inspection")
        print(f"{'='*80}\n")
        
        layer_count = 0
        for name, param in model.named_parameters():
            if layer_count >= num_layers:
                break
                
            print(f"Layer: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Size: {param.numel():,} parameters")
            print(f"  Data type: {param.dtype}")
            print(f"  Device: {param.device}")
            print(f"  Requires gradient: {param.requires_grad}")
            
            # Weight statistics
            if param.numel() > 0:
                print(f"  Min value: {param.min().item():.6f}")
                print(f"  Max value: {param.max().item():.6f}")
                print(f"  Mean value: {param.mean().item():.6f}")
                print(f"  Std dev: {param.std().item():.6f}")
            print()
            
            layer_count += 1
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")


def demonstrate_weight_sharing():
    """Demonstrate how model weights can be shared between instances."""
    print(f"\n{'='*80}")
    print("Demonstrating Weight Sharing")
    print(f"{'='*80}\n")
    
    model_name = "distilgpt2"
    
    # Load model
    print("Loading base model...")
    model1 = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create second instance that shares weights
    print("Creating second instance with shared weights...")
    model2 = model1
    
    # Verify weight sharing
    param1 = next(model1.parameters())
    param2 = next(model2.parameters())
    
    print(f"\nWeight sharing verification:")
    print(f"  Same memory location: {param1.data_ptr() == param2.data_ptr()}")
    print(f"  Model1 parameter address: {hex(param1.data_ptr())}")
    print(f"  Model2 parameter address: {hex(param2.data_ptr())}")
    
    # Memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model1.parameters())
    print(f"\nMemory usage:")
    print(f"  Parameters memory: {param_memory / 1e6:.2f} MB")
    print(f"  Shared between instances: Yes")


def demonstrate_weight_freezing():
    """Demonstrate freezing model weights."""
    print(f"\n{'='*80}")
    print("Demonstrating Weight Freezing")
    print(f"{'='*80}\n")
    
    model_name = "distilgpt2"
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Check initial trainable parameters
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nBefore freezing:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_before:,}")
    
    # Freeze all parameters except last layer
    print("\nFreezing all layers except the last one...")
    for name, param in model.named_parameters():
        if "lm_head" not in name:  # Keep only the output layer trainable
            param.requires_grad = False
    
    # Check trainable parameters after freezing
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nAfter freezing:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_after:,}")
    print(f"  Frozen parameters: {total_params - trainable_after:,}")
    print(f"  Reduction: {(1 - trainable_after/trainable_before)*100:.1f}%")


def demonstrate_weight_initialization():
    """Demonstrate different weight initialization strategies."""
    print(f"\n{'='*80}")
    print("Demonstrating Weight Initialization")
    print(f"{'='*80}\n")
    
    # Load configuration
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2  # Smaller model for demonstration
    
    print("Creating model from configuration...")
    model = AutoModelForCausalLM.from_config(config)
    
    print(f"Model created with random initialization")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Inspect a few weight values
    print("\nSample weight values from first layer:")
    first_param = next(model.parameters())
    print(f"  Shape: {first_param.shape}")
    print(f"  First 10 values: {first_param.flatten()[:10].tolist()}")
    print(f"  Mean: {first_param.mean().item():.6f}")
    print(f"  Std: {first_param.std().item():.6f}")


def main():
    """Main demonstration function."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MODEL WEIGHTS USAGE PATTERNS" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Initialize manager
    manager = ModelWeightsManager(cache_dir="./model_cache")
    
    # 1. Download and cache model
    print("\n" + "="*80)
    print("1. DOWNLOADING AND CACHING MODELS")
    print("="*80)
    tokenizer, model = manager.download_model("distilgpt2", precision="fp16")
    
    # 2. Inspect weights
    print("\n" + "="*80)
    print("2. INSPECTING MODEL WEIGHTS")
    print("="*80)
    manager.inspect_weights(model, num_layers=2)
    
    # 3. Save model locally
    print("\n" + "="*80)
    print("3. SAVING MODEL LOCALLY")
    print("="*80)
    save_path = "./saved_models/distilgpt2"
    manager.save_model_locally(model, tokenizer, save_path)
    
    # 4. Load from local path
    print("\n" + "="*80)
    print("4. LOADING MODEL FROM LOCAL PATH")
    print("="*80)
    try:
        local_tokenizer, local_model = manager.load_local_model(save_path)
        print("✓ Successfully loaded from local path")
    except Exception as e:
        print(f"Note: {e}")
    
    # 5. Demonstrate weight sharing
    print("\n" + "="*80)
    print("5. WEIGHT SHARING PATTERNS")
    print("="*80)
    demonstrate_weight_sharing()
    
    # 6. Demonstrate weight freezing
    print("\n" + "="*80)
    print("6. WEIGHT FREEZING FOR FINE-TUNING")
    print("="*80)
    demonstrate_weight_freezing()
    
    # 7. Demonstrate weight initialization
    print("\n" + "="*80)
    print("7. WEIGHT INITIALIZATION")
    print("="*80)
    demonstrate_weight_initialization()
    
    print("\n" + "="*80)
    print("DEMONSTRATIONS COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  • Models are cached locally after first download")
    print("  • Different precisions (fp32, fp16) affect memory usage")
    print("  • Weights can be frozen for efficient fine-tuning")
    print("  • Models can be saved and loaded from local directories")
    print("  • Weight sharing reduces memory when using same model multiple times")
    print("\nNext Steps:")
    print("  • Explore model quantization (int8, int4)")
    print("  • Learn about distributed model loading")
    print("  • Practice fine-tuning with frozen weights")
    print("  • Experiment with different model architectures")
    print()


if __name__ == "__main__":
    main()
