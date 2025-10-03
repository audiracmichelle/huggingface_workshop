# Huggingface Workshop: Opensource LLMs and Customization

This repository demonstrates how to explore and use opensource Large Language Models (LLMs) from Huggingface Hub, including loading model weights, performing inference, and understanding model architectures.

## 📚 Contents

- **explore_llms.ipynb**: Interactive Jupyter notebook with comprehensive examples
- **explore_models.py**: Python script for exploring models programmatically
- **requirements.txt**: Dependencies needed to run the examples

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. Clone this repository:
```bash
git clone https://github.com/audiracmichelle/huggingface_workshop.git
cd huggingface_workshop
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Quick Start

#### Using the Jupyter Notebook

```bash
jupyter notebook explore_llms.ipynb
```

The notebook includes:
- Environment setup and configuration
- Searching and filtering models on Huggingface Hub
- Loading model weights with different methods
- Text generation examples
- Model architecture inspection
- Performance optimization tips

#### Using the Python Script

```bash
python explore_models.py
```

This will:
- Check your environment setup
- Explore top models on Huggingface Hub
- Load and use GPT-2 for text generation
- Compare model sizes
- Demonstrate the pipeline API

## 📖 What You'll Learn

### 1. Exploring Available Models

Learn how to:
- Search for models by task (text-generation, classification, etc.)
- Filter models by popularity, downloads, and likes
- Get detailed information about specific models
- Understand model cards and documentation

### 2. Loading Model Weights

Multiple methods for loading models:
- **Simple loading**: `AutoModelForCausalLM.from_pretrained()`
- **With configuration**: Custom model configurations
- **Memory optimization**: Half-precision (fp16) and quantization
- **Local caching**: Reusing downloaded models

### 3. Using Models for Inference

- Basic text generation
- Controlling generation parameters (temperature, top_k, top_p)
- Using the pipeline API for simplified inference
- Batch processing for efficiency

### 4. Working with Different Model Types

Examples with:
- **GPT-2**: Causal language modeling
- **BERT**: Masked language modeling
- **DistilGPT-2**: Distilled models for efficiency
- Other popular architectures

## 🎯 Key Concepts

### Model Selection

Consider these factors when choosing a model:
- **Task alignment**: Match model capabilities to your needs
- **Model size**: Balance performance vs. resource constraints
- **License**: Check usage rights (especially for commercial use)
- **Community support**: Popular models have better documentation

### Model Weights

Understanding model weights:
- **Parameters**: Total number of trainable parameters
- **Precision**: fp32, fp16, int8 (affects size and speed)
- **Architecture**: Layer types and connections
- **Size on disk**: Storage requirements for downloaded models

### Performance Optimization

Tips for efficient model usage:
- Use GPU acceleration when available
- Load models in reduced precision (fp16)
- Implement batch processing
- Cache downloaded models locally
- Use distilled models for faster inference

## 📊 Example Models

Here are some recommended models to start with:

| Model | Size | Use Case | Good For |
|-------|------|----------|----------|
| `gpt2` | 124M | Text generation | Learning, experimentation |
| `distilgpt2` | 82M | Text generation | Resource-constrained environments |
| `bert-base-uncased` | 110M | Classification, NER | Understanding, embeddings |
| `gpt2-medium` | 355M | Text generation | Better quality generation |
| `EleutherAI/gpt-neo-125M` | 125M | Text generation | Open alternative to GPT |

## 🔧 Advanced Usage

### Loading Models with Custom Configuration

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("gpt2")
config.n_layer = 6  # Reduce number of layers
model = AutoModelForCausalLM.from_config(config)
```

### Using Half Precision for Memory Efficiency

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
```

### Generating with Custom Parameters

```python
outputs = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,      # Higher = more random
    top_k=50,             # Consider top 50 tokens
    top_p=0.95,           # Nucleus sampling
    do_sample=True,       # Enable sampling
    num_return_sequences=3  # Generate 3 variations
)
```

## 📚 Resources

- [Huggingface Model Hub](https://huggingface.co/models) - Browse thousands of models
- [Transformers Documentation](https://huggingface.co/docs/transformers) - Official docs
- [Huggingface Course](https://huggingface.co/course) - Free online course
- [Model Cards](https://huggingface.co/docs/hub/model-cards) - Understanding model documentation

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new examples
- Improve documentation
- Report issues
- Suggest new models to showcase

## 📝 License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Notes

- Model downloads can be large (several GB). Ensure sufficient disk space.
- First-time model loading will download weights from Huggingface Hub.
- GPU is recommended but not required for these examples.
- Some models may have specific license restrictions for commercial use.

## 🎓 Next Steps

After completing this workshop, consider:
1. **Fine-tuning**: Adapt models to your specific domain
2. **Deployment**: Serve models via API endpoints
3. **Optimization**: Explore quantization and pruning techniques
4. **Custom models**: Train your own models from scratch
5. **Multi-modal models**: Work with vision-language models

---

Happy exploring! 🚀
