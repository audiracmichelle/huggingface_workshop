# Discussion Guide — Section 1

## 1.2.4 Discussion: Datasets

### Find an image classification dataset on the Hub. How is it different from an image segmentation dataset?

**Example:** [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10) or [Food-101](https://huggingface.co/datasets/ethz/food101)

**Key differences:**

| | Image Classification | Image Segmentation |
|---|---|---|
| **Label** | One label per image (e.g., "cat") | One label per pixel (e.g., pixel-level mask) |
| **Output shape** | A single class or probability vector | A full-resolution mask matching the image dimensions |
| **Dataset structure** | `image` + `label` (integer or string) | `image` + `annotation` (mask image with class IDs per pixel) |
| **What it answers** | "What is in this image?" | "Where is each thing in this image?" |

**Guiding thought:** Classification tells you *what* is in the image. Segmentation tells you *where* each thing is, at pixel-level precision. Segmentation datasets are more expensive to create because every pixel needs a label, not just the whole image.

---

### Does the Hugging Face Datasets Hub follow the FAIR principles?

**Findable:**
- Yes. Datasets have unique identifiers (e.g., `restor/tcd`), are searchable by task/tag/keyword, and have metadata (size, features, license).
- The Hub acts like a registry with discovery tools (search, filters, SQL console).

**Accessible:**
- Yes. Datasets are downloadable via the `datasets` library or direct HTTP. Public datasets require no authentication.
- Gated datasets exist but access protocols are clearly documented.

**Interoperable:**
- Yes. Data is stored in Parquet (a widely supported columnar format). The `datasets` library is built on Apache Arrow, which interoperates with pandas, polars, and other tools.
- Feature types (Image, Audio, Value) are standardized across datasets.

**Reusable:**
- Mostly yes. Dataset cards document provenance, collection methodology, and licensing.
- Licenses vary — some datasets are CC-BY, others are more restrictive. Students should always check the license.

**Guiding thought:** The Hub does a strong job on F, A, and I. Reusability depends on how thoroughly the dataset card is filled out — some cards are sparse. This is a community effort and varies by dataset.

---

## 1.3.4 Discussion: Models

### For how long and with how many GPUs were each of the models trained?

From the [tcd-segformer-mit-b0 model card](https://huggingface.co/restor/tcd-segformer-mit-b0):

- **Hardware:** Single NVIDIA RTX 3090 (24 GB VRAM), 32-core machine with 64 GB RAM
- **Training time:** The smallest models (mit-b0) train in **under half a day**. The largest models take just over a day.
- **GPUs:** Just **1 GPU** for all models in the family

**Guiding thought:** Fine-tuning a SegFormer on a domain-specific dataset is very accessible — a single consumer GPU and less than a day of training. Compare this to training a foundation model like SAM from scratch (thousands of GPUs, weeks of training, millions of images).

### What is the carbon footprint of the training?

From the model card:

- **Carbon emitted:** ~5.44 kg CO2 equivalent per model (upper bound estimate for the largest model)
- **Estimated using:** [ML Impact Calculator](https://mlco2.github.io/impact#compute)
- **Caveat:** This doesn't include experimentation, failed runs, or cross-validation (actual cost was ~6x for cross-validation folds)

**Guiding thought:** 5.44 kg CO2 is roughly equivalent to driving a car 20 km. Fine-tuning is vastly cheaper in carbon terms than training from scratch. This matters for sustainable AI research — reusing pre-trained backbones and only fine-tuning the head is both computationally and environmentally efficient.
