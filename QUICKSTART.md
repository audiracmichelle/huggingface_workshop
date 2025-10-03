# Quick Start Guide

This guide will help you get started with exploring opensource LLMs on Huggingface in minutes.

## Prerequisites

- Python 3.8+
- 2GB+ free disk space (for model downloads)
- Internet connection

## Installation

```bash
# Clone the repository
git clone https://github.com/audiracmichelle/huggingface_workshop.git
cd huggingface_workshop

# Install dependencies
pip install -r requirements.txt
```

## Your First Model

### Option 1: Interactive Notebook (Recommended for Learning)

```bash
jupyter notebook explore_llms.ipynb
```

Then run the cells sequentially to learn about:
- Exploring available models
- Loading model weights
- Generating text
- Understanding model architectures

### Option 2: Python Script (Quick Demo)

```bash
python explore_models.py
```

This will automatically:
1. Show top models on Huggingface Hub
2. Load GPT-2 model
3. Generate sample text
4. Display model information

Expected output:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║               EXPLORING OPENSOURCE LLMs ON HUGGINGFACE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Environment Check
================================================================================
Transformers version: 4.x.x
PyTorch version: 2.x.x
CUDA available: True/False

Exploring Top Text Generation Models
================================================================================
...
```

### Option 3: Advanced Model Usage

```bash
python model_weights_usage.py
```

This demonstrates:
- Efficient model caching
- Saving/loading models locally
- Weight inspection
- Memory optimization techniques

## Simple Code Example

Here's a minimal example to get you started:

```python
from transformers import pipeline

# Create a text generation pipeline (downloads model automatically)
generator = pipeline('text-generation', model='gpt2')

# Generate text
result = generator("The future of AI is", max_length=30)
print(result[0]['generated_text'])
```

## Common Use Cases

### 1. Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)
text = tokenizer.decode(outputs[0])
print(text)
```

### 2. Exploring Models

```python
from huggingface_hub import list_models

# Find text generation models
models = list(list_models(task="text-generation", sort="downloads", limit=5))
for model in models:
    print(f"- {model.id}")
```

### 3. Model Information

```python
from huggingface_hub import model_info

info = model_info("gpt2")
print(f"Model: {info.id}")
print(f"Downloads: {info.downloads}")
print(f"Task: {info.pipeline_tag}")
```

## Troubleshooting

### Issue: Model download is slow
**Solution**: Models are cached after first download. Subsequent loads will be instant.

### Issue: Out of memory error
**Solution**: Use a smaller model like `distilgpt2` or load in fp16:
```python
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
```

### Issue: CUDA out of memory
**Solution**: Use CPU or a smaller model:
```python
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cpu")
```

### Issue: Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. **Experiment**: Try different models from [Huggingface Hub](https://huggingface.co/models)
2. **Learn**: Complete the Jupyter notebook cells
3. **Customize**: Modify generation parameters (temperature, top_k, top_p)
4. **Advanced**: Explore fine-tuning and model customization

## Resources

- 📖 [Full Documentation](README.md)
- 🤗 [Huggingface Model Hub](https://huggingface.co/models)
- 📚 [Transformers Docs](https://huggingface.co/docs/transformers)
- 🎓 [Free Course](https://huggingface.co/course)

## Getting Help

If you encounter issues:
1. Check the [Huggingface Documentation](https://huggingface.co/docs)
2. Search [Huggingface Forums](https://discuss.huggingface.co/)
3. Review the example notebooks in this repository

---

Happy coding! 🚀
