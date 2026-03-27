# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I want to create a Github pages website for the workshop:
    * For style and formatting, the website should use the following site as a guide https://astrostubbs.github.io/GenAI-for-Scholarship/site-map.html
    * For the content, the website should follow the agenda in section *Workshop Agenda* in the README:
        * for sections 1 and 3, utilize the flow and examples in the huggingface_workshop.ipynb and you can complement with this resource https://colab.research.google.com/drive/1N_rWko6jzGji3j_ayDR7ngT5lf4P8at_
        * for section 2, use the claude *Setup* in the README  and create a walkthrough a huggingface mcp
        * for section 4, figure our whether this is a good example of fine-tuning https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/models/segformer.py . If so, you can use it to ilustrate fine-tuning. Else, make up an example
    * Keep the index.html up to date

### Specs
    * make sure that an `image_train` and an `image_test` sample is pulled from the train and test splits
    * split the content like this:

```
Section 1 — Navigate the Hugging Face Ecosystem
  - 1.1 What is Hugging Face?
    - What are tasks?
  - 1.2 Working with Datasets on the Hub
    - 1.2.1 Step 1 — Filter datasets by task
    - 1.2.2 Step 2 — Explore data on the Hub
        - Understanding Parquet files
    - 1.2.3 Step 3 — Use the datasets library
        - Load a small dataset
        - Inspect splits, features, and shape
        - Pull samples and view images
        - When to use streaming=True
    - 1.2.4 Discussion
  - 1.3 Working with Models on the Hub
    - 1.3.1 Step 1 — Filter models by task
    - 1.3.2 Step 2 — Explore on the Hub
        - Understanding Safetensors files
    - 1.3.3 Step 3 — Use the transformers library: Manual Inference with Model + Processor
        - Load model and processor
        - Process inputs, run inference, post-process
        - Visualize the results
        - Inspect predicted classes
    - 1.3.4 Discussion
  - 1.5 Additional Resources

  Section 3 — Run Models in Minutes
  - 3.1 Get data samples with datasets
  - 3.2 Perform Inference with transformers 
  - 3.3 Compare NVIDIA SegFormer and TCD-SegFormer on image_test
    - 3.3.1 Key differences                                                                                                                                                 
  - 3.4 Discussion
```

In 1.2.4 ask the students to identify an image classification dataset and explain how it is different to an image segmentation dataset

## Setup

```bash
pip install -r requirements.txt
```

Environment variables for Claude Code access via Harvard billing:
```bash
export ANTHROPIC_BEDROCK_BASE_URL=https://apis.huit.harvard.edu/ais-bedrock-llm/v2
export CLAUDE_CODE_USE_BEDROCK=1
export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1
```

## Notebook

- **`huggingface_workshop.ipynb`** — Main workshop notebook. Four-part progression: 
HF ecosystem → 
Datasets (`datasets` library) → 
Models (`transformers` library) → 
Community. 
Uses the Restor tree cover dataset (`restor/tcd`, `restor/tcd-nc`) and SegFormer models for image segmentation.

### Considerations for website walkthrough 
    - Each code cell starts with a descriptive docstring in triple quotes
    - Educational tone: code includes comments and explanations for workshop participants
    - Avoid complex abstractions that would obscure learning objectives

Feel free to do the following if it makes sense
    - Use `datasets.load_dataset()` with `streaming=True` for large datasets (e.g., `restor/tcd`), `streaming=False` for small ones (e.g., `restor/tcd-nc`)
    - Use segmentation masks `tab20` colormap with matplotlib overlay (`alpha=0.5`)

Incorporate additional notes to:
    - Explain what the `Auto*` pattern is: `AutoModelForSemanticSegmentation`, `AutoImageProcessor`
    - Explain why to prefer `pipeline()` for simplified inference over manual model/processor workflows
    - Explain why models/processors with `from_pretrained()`; run inference inside `torch.no_grad()`
    - Explain why images are PIL objects; use `.resize((512, 512))` for visualization

## Think of a way to test SAM on an image of the student

Try it yourself — Use the ID of an image stored in your Google Drive and run the SlimSAM-uniform-77 segmenter:

sam_pipeline = pipeline("image-segmentation", model="Zigeng/SlimSAM-uniform-77")
sam_pipeline("https://drive.google.com/uc?export=view&id=<your_image_id>")
How many masks and which labels do the results have?

"""
  Try it yourself: run SlimSAM-uniform-77 on your own image
"""
#sam_pipeline = pipeline("image-segmentation", model="Zigeng/SlimSAM-uniform-77")
#sam_pipeline("https://drive.google.com/uc?export=view&id=<your_image_id>")