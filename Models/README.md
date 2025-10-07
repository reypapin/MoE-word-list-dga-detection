# Expert Models for Wordlist-Based DGA Detection

This directory documents the seven expert models evaluated in our research. **All trained models are hosted on HuggingFace** for easy access and reproducibility.

---

## 🚀 Quick Access

**All models are available at:**
🔗 **https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models**

Download directly from HuggingFace instead of cloning this repository to save bandwidth and storage space.

---

## 📦 Available Models

### 1. ModernBERT Expert ⭐ (OPTIMAL MODEL)

**HuggingFace:** [`models/modernbert-wordlist-expert/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/modernbert-wordlist-expert)

**Performance:**
- **F1-Score (Known):** 86.7% ± 3.0%
- **F1-Score (Unknown):** 80.9% ± 4.5%
- **Inference Time:** 26ms (Tesla T4 GPU)
- **Throughput:** ~38,000 domains/second

**Description:**
Fine-tuned ModernBERT model trained exclusively on 8 wordlist-based DGA families. Identified as the optimal expert through systematic evaluation, balancing accuracy, generalization, and inference speed.

**Base Model:** `answerdotai/ModernBERT-base`

**Training:**
- Dataset: 160,000 balanced samples (train_wl.csv)
- Families: charbot, deception, gozi, manuelita, matsnu, nymaim, rovnix, suppobox
- Epochs: 3
- Batch size: 16

**Load from HuggingFace:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/modernbert-wordlist-expert"
)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
```

**Local Training Notebook:** `../Notebook/ModernBERT_base_DGA_Word.ipynb`

---

### 2. ModernBERT Generalist (Baseline)

**HuggingFace:** [`models/modernbert-generalist-54f/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/modernbert-generalist-54f)

**Performance:**
- **F1-Score (Known):** 79.2% ± 3.5%
- **F1-Score (Unknown):** 62.1% ± 5.2%
- **Inference Time:** 27ms (Tesla T4 GPU)

**Description:**
ModernBERT trained on 54 diverse DGA families (wordlist + algorithmic). Serves as baseline to demonstrate the advantage of specialist training over generalist approaches.

**Key Finding:** Specialist approach improves F1-score by **+9.4%** on known families and **+30.2%** on unseen families.

**Training:**
- Dataset: 1,080,000 samples (train_1M.csv)
- Families: 54 diverse DGA types
- Approach: Multi-family generalist

**Load from HuggingFace:**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/modernbert-generalist-54f"
)
```

**Local Training Notebook:** `../Notebook/ModernBERT_base_DGA_54F.ipynb`

---

### 3. Gemma 3 4B with LoRA Adapters

**HuggingFace:** [`models/gemma-3-4b-lora/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/gemma-3-4b-lora)

**Performance:**
- **Precision:** 95.4% (exceptional)
- **Recall:** 66.5% (moderate)
- **F1-Score:** 78.6%
- **Inference Time:** ~650ms (too slow for real-time)

**Description:**
Large Language Model fine-tuned with LoRA (Low-Rank Adaptation) on wordlist-based DGAs. Achieves highest precision but suffers from low recall and slow inference.

**Base Model:** `google/gemma-3-4b-it`

**Architecture:**
- LoRA adapters only (95 MB)
- Requires base model (4B parameters)
- 8-bit quantization for efficiency

**Load from HuggingFace:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    device_map="auto",
    load_in_8bit=True
)
model = PeftModel.from_pretrained(
    base_model,
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/gemma-3-4b-lora"
)
```

**Local Training Notebook:** `../Notebook/Train_Gemma3_4B_DGA_WordList.ipynb`

---

### 4. LLaMA 3.2 3B with LoRA Adapters

**HuggingFace:** [`models/llama-3.2-3b-lora/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/llama-3.2-3b-lora)

**Performance:**
- **F1-Score (Known):** 81.4%
- **F1-Score (Unknown):** 74.8%
- **Inference Time:** ~680ms

**Description:**
LLaMA 3.2 3B fine-tuned with LoRA adapters. Similar characteristics to Gemma: good accuracy but impractical inference time for production.

**Base Model:** `meta-llama/Llama-3.2-3B`

**Architecture:**
- LoRA adapters (110 MB)
- Requires base model (3B parameters)
- Supports 8-bit quantization

**Load from HuggingFace:**
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    load_in_8bit=True
)
model = PeftModel.from_pretrained(
    base_model,
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/llama-3.2-3b-lora"
)
```

**Local Training Notebook:** `../Notebook/Train_llama3B_DGA_WordList.ipynb`

---

### 5. DomBertUrl

**HuggingFace:** [`models/dombert-url/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/dombert-url)

**Performance:**
- **F1-Score (Known):** 81.2%
- **F1-Score (Unknown):** 84.6% (best generalization)
- **Inference Time:** 28ms

**Description:**
Domain-specialized BERT variant with LoRA adapters. Achieves the **best generalization performance** on unseen DGA families (84.6%), making it particularly valuable for zero-shot detection scenarios.

**Base Model:** Domain-pretrained BERT

**Architecture:**
- LoRA adapters (1.4 MB)
- Lightweight and efficient

**Load from HuggingFace:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load with LoRA adapters
model = AutoModelForSequenceClassification.from_pretrained(
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/dombert-url"
)
```

**Local Training Notebook:** `../Notebook/DomUrlBert.ipynb`

---

### 6. CNN Wordlist

**HuggingFace:** [`models/cnn-wordlist/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/cnn-wordlist)

**Performance:**
- **F1-Score (Known):** 78.9%
- **F1-Score (Unknown):** 72.1%
- **Inference Time:** 15ms (very fast)
- **Throughput:** ~66,000 domains/second

**Description:**
Convolutional Neural Network with custom architecture for wordlist pattern detection. Offers the fastest inference among deep learning models, suitable for ultra-high throughput scenarios.

**Architecture:**
- Custom CNN layers
- Character-level embeddings
- Lightweight (76 KB)

**Load from HuggingFace:**
```python
import torch

model = torch.load(
    "hf://Reynier/moe-wordlist-dga-models/models/cnn-wordlist/dga_cnn_model_wl.pth"
)
```

**Local Training Notebook:** `../Notebook/CNN_Patron_WL.ipynb`

---

### 7. FANCI (Random Forest)

**HuggingFace:** [`models/fanci/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/fanci)

**Performance:**
- **F1-Score (Known):** 77.3%
- **F1-Score (Unknown):** 68.5%
- **Inference Time:** <1ms (extremely fast)
- **Throughput:** >100,000 domains/second

**Description:**
Traditional machine learning approach using Random Forest with hand-crafted features (n-grams, entropy, length, etc.). Fastest inference with no GPU requirement, ideal for resource-constrained environments.

**Architecture:**
- Random Forest classifier
- Feature engineering: n-grams, statistical features
- No GPU required
- Size: 794 MB (includes dictionaries)

**Load from HuggingFace:**
```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Reynier/moe-wordlist-dga-models",
    filename="models/fanci/fanci_dga_detector_20250618_164818.joblib"
)
model = joblib.load(model_path)
```

**Local Training Notebook:** `../Notebook/FANCI.ipynb`

---

### 8. LABin

**HuggingFace:** [`models/labin/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models/labin)

**Performance:**
- **F1-Score (Known):** 75.6%
- **F1-Score (Unknown):** 70.2%
- **Inference Time:** 18ms

**Description:**
Hybrid model combining linguistic analysis with attention mechanisms. Focuses on extracting meaningful subword patterns from wordlist-based DGAs.

**Architecture:**
- Keras/TensorFlow implementation
- Attention-based feature extraction
- Size: 8.1 MB

**Load from HuggingFace:**
```python
from tensorflow import keras
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Reynier/moe-wordlist-dga-models",
    filename="models/labin/LABin_best_model_2025-05-30_15_26_47.keras"
)
model = keras.models.load_model(model_path)
```

**Local Training Notebook:** `../Notebook/Labin_wl.ipynb`

---

## 📊 Model Comparison Summary

| Model | Known F1 | Unknown F1 | Inference (ms) | Size | GPU Required |
|-------|----------|------------|----------------|------|--------------|
| **ModernBERT Expert** ⭐ | **86.7%** | **80.9%** | **26** | 575 MB | Yes |
| ModernBERT Generalist | 79.2% | 62.1% | 27 | 575 MB | Yes |
| Gemma 3 4B LoRA | 78.6% | 73.2% | 650 | 95 MB* | Yes |
| LLaMA 3.2 3B LoRA | 81.4% | 74.8% | 680 | 110 MB* | Yes |
| DomBertUrl | 81.2% | **84.6%** | 28 | 1.4 MB* | Yes |
| CNN | 78.9% | 72.1% | **15** | 76 KB | Yes |
| FANCI | 77.3% | 68.5% | **<1** | 794 MB | No |
| LABin | 75.6% | 70.2% | 18 | 8.1 MB | Yes |

\* LoRA adapters only; requires separate base model download

---

## 🎯 Model Selection Guide

### Best Overall Performance
→ **ModernBERT Expert** (86.7% F1, 26ms inference)

### Best Generalization to Unknown DGAs
→ **DomBertUrl** (84.6% F1 on unseen families)

### Fastest Inference (CPU)
→ **FANCI** (<1ms, no GPU)

### Fastest Deep Learning Model
→ **CNN** (15ms, GPU)

### Highest Precision (Low False Positives)
→ **Gemma 3 4B** (95.4% precision, but slow)

### Resource Constrained Environment
→ **FANCI** (CPU-only, traditional ML)

---

## 📥 Download Instructions

### Option 1: Download Individual Model from HuggingFace

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download specific model
huggingface-cli download Reynier/moe-wordlist-dga-models \
  --include "models/modernbert-wordlist-expert/*" \
  --local-dir ./models
```

### Option 2: Load Directly in Python

```python
from transformers import AutoModelForSequenceClassification

# Automatically downloads and caches
model = AutoModelForSequenceClassification.from_pretrained(
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/modernbert-wordlist-expert"
)
```

### Option 3: Clone Entire Model Repository

```bash
git lfs install
git clone https://huggingface.co/Reynier/moe-wordlist-dga-models
cd moe-wordlist-dga-models/models/
```

---

## 🔧 Local Training

All training notebooks are available in the `Notebook/` directory:

```bash
cd ../Notebook/
jupyter notebook ModernBERT_base_DGA_Word.ipynb
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

See individual notebook READMEs for specific dependencies.

---

## 📚 Datasets

Training and evaluation datasets are available on HuggingFace:

- **Training Set:** [`datasets/train_wl.csv`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/datasets) (160K samples)
- **Test Sets (Known):** [`datasets/test-known/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/datasets/test-known) (8 families)
- **Test Sets (Generalization):** [`datasets/test-generalization/`](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/datasets/test-generalization) (3 families)

---

## 📖 Citation

If you use these models in your research, please cite:

```bibtex
@article{leyva2025expert,
  title={Expert Selection for Wordlist-Based DGA Detection: A Systematic Evaluation},
  author={Leyva La O, Reynier and Catania, Carlos A. and Gonzalez, Rodrigo},
  journal={Under Review},
  year={2025}
}
```

---

## 📞 Support

For questions or issues:
- **HuggingFace Repository:** https://huggingface.co/Reynier/moe-wordlist-dga-models
- **GitHub Repository:** https://github.com/reypapin/MoE-word-list-dga-detection
- **Contact:** rleyvalao@mendoza-conicet.gob.ar

---

**Note:** This directory no longer contains model weights to reduce repository size. All models are hosted on HuggingFace for efficient distribution and version control.
