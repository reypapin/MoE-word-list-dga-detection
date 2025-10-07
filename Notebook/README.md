# Training & Evaluation Notebooks

This directory contains Jupyter notebooks for training, evaluating, and comparing the seven expert models for wordlist-based DGA detection.

---

## 🔗 Quick Links

- **Trained Models:** [HuggingFace](https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models)
- **Datasets:** [`../Dataset/`](../Dataset/)
- **Results:** [`../Result_csv/`](../Result_csv/)
- **Paper:** Under Review

---

## 📂 Directory Structure

```
Notebook/
├── README.md                           # This file
├── ModernBERT_base_DGA_Word.ipynb     # ⭐ Train optimal expert (161 KB)
├── ModernBERT_base_DGA_54F.ipynb      # Train generalist baseline (110 KB)
├── DomUrlBert.ipynb                    # Train DomBertUrl model (114 KB)
├── Train_Gemma3_4B_DGA_WordList.ipynb # Train Gemma LoRA (289 KB)
├── Train_llama3B_DGA_WordList.ipynb   # Train LLaMA LoRA (105 KB)
├── Test_Gemma3_4B_DGA_Last.ipynb      # Test Gemma model (200 KB)
├── Test__llama3B_DGA.ipynb            # Test LLaMA model (43 KB)
├── CNN_Patron_WL.ipynb                # Train CNN model (27 KB)
├── FANCI.ipynb                         # Train FANCI RF (581 KB)
└── Labin_wl.ipynb                      # Train LABin hybrid (70 KB)
```

**Total Size:** ~1.7 MB

---

## 📚 Notebook Index

### 🏆 ModernBERT (Optimal Expert)

#### **ModernBERT_base_DGA_Word.ipynb** ⭐

**Purpose:** Train the optimal expert model on 8 wordlist-based DGA families

**Key Features:**
- Fine-tunes `answerdotai/ModernBERT-base` on 160K balanced samples
- Two-phase evaluation (known + unknown families)
- Achieves **86.7% F1** on known families, **80.9%** on unknown
- **26ms inference** time on Tesla T4 GPU
- Complete training pipeline with hyperparameter search

**Output:**
- Model checkpoint: `../Models/modernbert-dga-detector/`
- Metrics: `../Result_csv/ModernBERT_DGA_WL_metrics_summary.csv`
- Training time: ~2-3 hours on T4 GPU

**Start here for reproducing paper results!**

---

#### **ModernBERT_base_DGA_54F.ipynb**

**Purpose:** Train generalist baseline for comparison

**Key Features:**
- ModernBERT trained on 54 diverse DGA families (1.08M samples)
- Used to validate specialist vs. generalist approach
- Performance: **79.2% F1** (known), **62.1% F1** (unknown)
- Demonstrates **+9.4% improvement** from specialist training

**Output:**
- Model checkpoint: `../Models/modernbert-dga-detector-54familias/`
- Training time: ~4-5 hours on T4 GPU

**Key Finding:** Specialist training significantly outperforms generalist approach on wordlist-based DGAs.

---

### 🤖 Large Language Models (LLMs)

#### **Train_Gemma3_4B_DGA_WordList.ipynb**

**Purpose:** Fine-tune Gemma 3 4B with LoRA adapters

**Key Features:**
- Parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation)
- Exceptional precision: **95.4%** (but lower recall: 66.5%)
- 8-bit quantization for memory efficiency
- Inference time: ~650ms (too slow for real-time)

**Output:**
- LoRA adapters: `../Models/gemma_dga_detector/` (95 MB)
- Requires base model: `google/gemma-3-4b-it`
- Training time: ~6-8 hours on A100/V100

**Requirements:**
- GPU: 16-20 GB VRAM
- Libraries: `peft`, `bitsandbytes`, `accelerate`

---

#### **Train_llama3B_DGA_WordList.ipynb**

**Purpose:** Fine-tune LLaMA 3.2 3B with LoRA adapters

**Key Features:**
- QLoRA (4-bit quantization) for memory-efficient training
- F1-Score: **81.4%** (known), **74.8%** (unknown)
- Inference time: ~680ms
- Precision-recall trade-off analysis

**Output:**
- LoRA adapters: `../Models/llama3.2_3B_dectector_dga/` (110 MB)
- Requires base model: `meta-llama/Llama-3.2-3B`
- Training time: ~4-6 hours on T4 (with quantization)

**Requirements:**
- GPU: 12-16 GB VRAM
- Access to LLaMA weights (via HuggingFace)

---

#### **Test_Gemma3_4B_DGA_Last.ipynb**

**Purpose:** Comprehensive evaluation of trained Gemma model

**Key Features:**
- Two-phase evaluation protocol
- Per-family performance breakdown
- Inference time profiling
- False positive/negative case study

**Input:** Trained Gemma model from HuggingFace or local checkpoint

---

#### **Test__llama3B_DGA.ipynb**

**Purpose:** Evaluate trained LLaMA model

**Key Features:**
- 8-bit quantized inference evaluation
- Latency-accuracy trade-off analysis
- Production deployment feasibility study

**Input:** Trained LLaMA model from HuggingFace or local checkpoint

---

### 🔬 Specialized Transformers

#### **DomUrlBert.ipynb**

**Purpose:** Train domain-specialized BERT with LoRA

**Key Features:**
- Leverages domain-specific pretraining
- **Best generalization:** **84.6% F1** on unknown families
- LoRA adapters: lightweight (1.4 MB)
- Inference time: **28ms**

**Output:**
- LoRA adapters: `../Models/DomBertUrl/`
- Training time: ~1-2 hours on T4 GPU

**Highlight:** Achieves highest F1-score on unseen DGA families, demonstrating superior transfer learning.

---

### ⚡ High-Speed Models

#### **CNN_Patron_WL.ipynb**

**Purpose:** Train character-level CNN for fast inference

**Key Features:**
- Character embeddings + convolutional layers
- **Fastest deep learning model:** **15ms** inference
- F1-Score: **78.9%** (known), **72.1%** (unknown)
- Throughput: ~66,000 domains/second

**Output:**
- Model weights: `../Models/dga_cnn_model_wl/dga_cnn_model_wl.pth` (76 KB)
- Training time: <30 minutes on T4 GPU

**Use Case:** Ultra-high throughput scenarios where speed > accuracy.

---

#### **FANCI.ipynb**

**Purpose:** Train Random Forest with hand-crafted features

**Key Features:**
- Traditional ML baseline (no GPU required)
- Feature engineering: n-grams, entropy, length, character frequency
- **Fastest inference:** **<1ms** (CPU-only)
- F1-Score: **77.3%** (known), **68.5%** (unknown)

**Output:**
- Model + vectorizers: `../Models/FANCI_model/` (~794 MB with dictionaries)
- Training time: <15 minutes on CPU

**Use Case:** Resource-constrained environments, CPU-only deployment.

---

#### **Labin_wl.ipynb**

**Purpose:** Train hybrid linguistic-attention model

**Key Features:**
- Combines character-level and semantic features
- Keras/TensorFlow implementation
- F1-Score: **75.6%** (known), **70.2%** (unknown)
- Inference time: **18ms**

**Output:**
- Keras model: `../Models/LABIN/` (8.1 MB)
- Training time: ~1 hour on T4 GPU

---

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n dga-detection python=3.10
conda activate dga-detection

# Core dependencies
pip install torch>=2.0.0 transformers>=4.35.0
pip install scikit-learn>=1.3.0 pandas>=2.0.0 numpy>=1.24.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0 jupyter>=1.0.0

# For LLM training (optional - requires GPU ≥16GB VRAM)
pip install peft>=0.5.0 bitsandbytes>=0.41.0 accelerate>=0.24.0
```

### Step-by-Step Workflow

#### 1. Data Preparation

Ensure datasets are in `../Dataset/`:
```
../Dataset/
├── train/
│   ├── train_wl.csv      # 160K samples (expert training)
│   └── train_1M.csv      # 1.08M samples (generalist training)
├── test-known/           # 8 DGA families (724K samples)
└── test-generalization/  # 3 unseen families (13.6K samples)
```

#### 2. Train Optimal Expert

```bash
# Start Jupyter
jupyter notebook

# Open and run
ModernBERT_base_DGA_Word.ipynb
```

**Execution steps:**
1. Load training data (`train_wl.csv`)
2. Initialize ModernBERT from HuggingFace
3. Fine-tune for 3 epochs (~2-3 hours)
4. Evaluate on test-known (Phase 1)
5. Evaluate on test-generalization (Phase 2)
6. Save model checkpoint

**Expected Results:**
- Phase 1 (Known): 86.7% ± 3.0% F1
- Phase 2 (Unknown): 80.9% ± 4.5% F1
- Inference: ~26ms per domain

#### 3. Train Alternative Models (Optional)

```bash
# For comparison with other architectures
jupyter notebook DomUrlBert.ipynb          # Best generalization
jupyter notebook CNN_Patron_WL.ipynb        # Fastest inference
jupyter notebook FANCI.ipynb                # CPU-only baseline
```

#### 4. Train Generalist Baseline (Optional)

```bash
# To validate specialist vs. generalist approach
jupyter notebook ModernBERT_base_DGA_54F.ipynb
```

**Use `train_1M.csv`** (1.08M samples, 54 families)

---

## 💻 Hardware Requirements

| Model | GPU Memory | Training Time | Inference | Recommended GPU |
|-------|-----------|---------------|-----------|----------------|
| **ModernBERT Expert** ⭐ | 8-12 GB | 2-3 hours | 26ms | Colab T4, RTX 3080 |
| ModernBERT Generalist | 8-12 GB | 4-5 hours | 27ms | Colab T4, RTX 3080 |
| DomBertUrl | 6-8 GB | 1-2 hours | 28ms | Colab T4, GTX 1080 Ti |
| Gemma 3 4B | 16-20 GB | 6-8 hours | 650ms | Colab A100, RTX 4090 |
| LLaMA 3.2 3B | 12-16 GB | 4-6 hours | 680ms | Colab T4 (quantized) |
| CNN | 2-4 GB | <30 min | 15ms | Any GPU, CPU ok |
| FANCI | CPU only | <15 min | <1ms | Any CPU |
| LABin | 4-6 GB | ~1 hour | 18ms | Colab T4, GTX 1660 |

> **Note:** All paper experiments used **NVIDIA Tesla T4 GPUs** on Google Colab for reproducibility.

---

## 📊 Expected Outputs

### Model Checkpoints

Saved to `../Models/{model_name}/`:

```
Models/modernbert-dga-detector/
├── config.json                  # Model architecture
├── model.safetensors           # Trained weights (575 MB)
├── tokenizer.json              # Tokenization config
├── tokenizer_config.json
├── special_tokens_map.json
└── training_args.bin           # Hyperparameters
```

### Performance Metrics

Saved to `../Result_csv/`:

```csv
Model,Precision_Known,Recall_Known,F1_Known,FPR_Known,Precision_Unknown,Recall_Unknown,F1_Unknown,FPR_Unknown,Inference_ms
ModernBERT,89.7±4.1,86.6±3.1,86.7±3.0,9.0±3.8,89.0±4.4,75.5±5.6,80.9±4.5,9.1±4.1,26
```

### Visualizations

Generated inline in notebooks:
- Training/validation loss curves
- F1-score distributions (boxplots)
- Confusion matrices per family
- ROC curves and precision-recall curves
- Inference time vs. accuracy scatter plots

---

## 🔬 Reproducibility

All notebooks implement:

1. **Fixed Random Seeds**
   ```python
   import random
   import numpy as np
   import torch

   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   ```

2. **Deterministic Algorithms**
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

3. **Stratified Sampling**
   - Maintains class balance across train/val/test splits
   - Ensures representative family distribution

4. **Documented Hyperparameters**
   - All hyperparameters clearly specified in config cells
   - Grid search ranges and final selections documented

### Reproducing Paper Results

1. Use **identical dataset** (`../Dataset/train/train_wl.csv`)
2. Follow **hyperparameters** in notebook config cells
3. Run on **Tesla T4 GPU** (or adjust batch size for other GPUs)
4. Use **30 randomized batches** for evaluation (matching paper protocol)

---

## 🛠️ Customization

### Adding New DGA Families

```python
# In data loading cell, add new families
new_families = ['family_name_1', 'family_name_2']

# Update family distribution
families_to_load = {
    'charbot': 10000,
    'deception': 10000,
    # ... existing families ...
    'family_name_1': 10000,  # Add new
    'family_name_2': 10000,  # Add new
}

# Update evaluation to include in unknown set
unknown_families = ['bigviktor', 'ngioweb', 'pizd', 'family_name_1']
```

### Hyperparameter Tuning

```python
# Key hyperparameters to adjust
training_args = TrainingArguments(
    learning_rate=2e-5,      # Try: 1e-5, 2e-5, 5e-5
    per_device_train_batch_size=16,  # Adjust for GPU memory
    num_train_epochs=3,      # Increase if underfitting
    warmup_steps=500,        # ~10% of total steps
    weight_decay=0.01,       # L2 regularization
    logging_steps=100,       # Logging frequency
)
```

### Model Architecture

```python
# For transformers - adjust capacity
config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
config.num_hidden_layers = 12  # Default: 12 (try 6, 8, 10)
config.hidden_size = 768       # Default: 768 (try 512, 1024)

# For CNN - modify filters
filters = [128, 256, 512]     # Adjust capacity
kernel_sizes = [3, 4, 5]      # N-gram sizes
dropout = 0.5                  # Regularization
```

---

## 📈 Evaluation Protocol

### Phase 1: Known Families

**Dataset:** `../Dataset/test-known/` (8 families, 724K samples)

**Protocol:**
1. Random sample 100 domains per family
2. Random sample 100 benign domains (Tranco)
3. Shuffle all 900 domains
4. Classify and compute metrics
5. Repeat 30 times for statistical confidence

**Metrics:**
- Precision, Recall, F1-Score
- False Positive Rate (FPR)
- Inference time per domain

### Phase 2: Generalization (Unknown Families)

**Dataset:** `../Dataset/test-generalization/` (3 families, 13.6K samples)

**Protocol:**
- Test on all available samples (no random sampling)
- Compare performance against Phase 1
- Measure generalization gap

**Expected Performance Drop:**
- ModernBERT: ~6% F1 drop (acceptable)
- FANCI: ~9% F1 drop (poor generalization)

---

## 📖 Additional Resources

### Documentation

- **Transformers:** https://huggingface.co/docs/transformers
- **PyTorch:** https://pytorch.org/tutorials/
- **scikit-learn:** https://scikit-learn.org/stable/
- **PEFT (LoRA):** https://huggingface.co/docs/peft

### Related Files

- **Datasets:** [`../Dataset/README.md`](../Dataset/README.md)
- **Models:** [`../Models/README.md`](../Models/README.md)
- **Results:** [`../Result_csv/`](../Result_csv/)
- **Paper:** Under Review

### Pre-trained Models

All trained models available on HuggingFace:
https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/models

Download and use without training:
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/modernbert-wordlist-expert"
)
```

---

## ❓ Troubleshooting

### Out of Memory (OOM) Errors

```python
# Reduce batch size
per_device_train_batch_size = 8  # Instead of 16

# Enable gradient accumulation
gradient_accumulation_steps = 2  # Effective batch size: 8 * 2 = 16

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

### CUDA Not Available

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Force CPU training (slower)
device = "cpu"
model = model.to(device)
```

### Slow Training

```python
# Enable mixed precision (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
# Use in training loop

# Or with Trainer API
training_args = TrainingArguments(
    fp16=True,  # Enable mixed precision
    dataloader_num_workers=4,  # Parallel data loading
)
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade torch transformers
pip install --upgrade peft bitsandbytes accelerate

# Check versions
pip list | grep torch
pip list | grep transformers
```

---

## 📞 Support

For questions or issues:

1. **Check inline comments** in notebooks for detailed explanations
2. **Review paper methodology** (Section IV) for theoretical background
3. **Examine expected outputs** in `../Result_csv/` for format reference
4. **Open GitHub issue** with error messages and environment details

**Contact:**
- **Author:** Reynier Leyva La O
- **Email:** rleyvalao@mendoza-conicet.gob.ar
- **GitHub:** https://github.com/reypapin/MoE-word-list-dga-detection
- **HuggingFace:** https://huggingface.co/Reynier/moe-wordlist-dga-models

---

## 📖 Citation

If you use these notebooks in your research, please cite:

```bibtex
@article{leyva2025expert,
  title={Expert Selection for Wordlist-Based DGA Detection: A Systematic Evaluation},
  author={Leyva La O, Reynier and Catania, Carlos A. and Gonzalez, Rodrigo},
  journal={Under Review},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- **Compute:** NVIDIA Tesla T4 GPUs provided by Google Colab
- **Datasets:** DGArchive, 360 Netlab, UMUDga, Tranco
- **Base Models:** ModernBERT (Answer.AI), Gemma (Google), LLaMA (Meta)
- **Institution:** CONICET Argentina

---

**Last Updated:** October 2025
