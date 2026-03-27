# huggingface_workshop
This repo contains the materials for a hands-on workshop on **Hugging Face** datasets and models.

## Workshop Agenda

1. **Navigate the Hugging Face ecosystem**
   - Explore Spaces, tasks, datasets, and models on the Hub
   - Read dataset and model cards
   - Understand why Hugging Face formats are well suited for open source
     - Parquet for datasets
     - Safetensors for model weights
   - Work with the `datasets` and `transformers` libraries

2. **Use Hugging Face MCP with Claude Code or Codex**
   - Access Claude through a **Harvard billing account**
   - Access Codex with your **HUID**
   - Explore MCP-powered workflows

3. **Run models in minutes**
   - Prepare the Restor dataset with `datasets`
   - Perform inference with `transformers`
   - Compare NVIDIA SegFormer and TCD-SegFormer on a test sample

4. **Build an end-to-end training loop**
   - Fine-tune NVIDIA SegFormer on the Restor dataset
   - Track evaluation metrics
   - Briefly explore SAM (Segment Anything Model) as a modern foundation model for segmentation


<!-- 5. Ship & share reproducible results: save/version artifacts and publish your model (and supporting assets) so others can reuse your work. -->

## Setup

```
export ANTHROPIC_BEDROCK_BASE_URL=https://apis.huit.harvard.edu/ais-bedrock-llm/v2
export ANTHROPIC_API_KEY=FaVxoM0WlGxnifFgETTzVlFixjieOhHkblEkcNhATA9hfUGn
export ANTHROPIC_SMALL_FAST_MODEL=us.anthropic.claude-opus-4-5-20251101-v1:0
export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1
export CLAUDE_CODE_USE_BEDROCK=1
```

## Materials
* [Website](https://audiracmichelle.github.io/huggingface_workshop/)
* [Follow Along Notebook](./follow_along.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/audiracmichelle/huggingface_workshop/blob/main/follow_along.ipynb)
* [Notebook](./huggingface_workshop.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/audiracmichelle/huggingface_workshop/blob/main/huggingface_workshop.ipynb)

## Requirements
If you want to run the notebooks locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
