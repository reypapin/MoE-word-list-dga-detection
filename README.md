# Mixture of Experts for Wordlist-Based DGA Detection

**Systematic evaluation of expert models for detecting wordlist-based Domain Generation Algorithms (DGAs) using transformer, CNN, and traditional machine learning approaches.**


[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models%20%26%20Datasets-yellow)](https://huggingface.co/Reynier/moe-wordlist-dga-models)
[![License](https://img.shields.io/badge/License-Research-blue)]()

---

## 📖 Overview

Domain Generation Algorithms (DGAs) have evolved beyond pseudorandom patterns to use **wordlist-based variants** that generate linguistically coherent domains, evading conventional detection methods. This research provides a **systematic evaluation of seven expert models** specifically targeting wordlist-based DGA detection within a Mixture of Experts (MoE) architecture.

### Key Results

Our evaluation identifies **ModernBERT as the optimal expert model**, achieving:

- **86.7% F1-score** on known DGA families
- **80.9% F1-score** on unseen families (generalization)
- **26ms inference time** (Tesla T4 GPU)
- **~38 domains/second** throughput

This represents a **+9.4% improvement** over generalist approaches on familiar variants and **+30.2% on unseen families**.

---

## 🎯 Research Contributions

1. **Systematic Evaluation Framework**: Methodology for evaluating MoE expert models targeting wordlist-based DGAs
2. **Comprehensive Empirical Analysis**: Rigorous evaluation across 7 models and 11 DGA families
3. **Two-Phase Evaluation Protocol**: Measures both in-distribution performance and zero-shot generalization
4. **Optimal Expert Identification**: ModernBERT characterized as best balance of accuracy, speed, and generalization
5. **Reproducible Research**: Complete code, models, datasets, and results publicly available

---

## 📂 Repository Structure

```
MoE-word-list-dga-detection/
├── Models/                    # Expert model documentation (models hosted on HuggingFace)
│   └── README.md              # Detailed descriptions of 7 expert models
│
├── Dataset/                   # Training and evaluation datasets
│   ├── train/                 # Training data (160K domains, 8 families)
│   ├── test-known/            # Test sets for known families
│   ├── test-generalization/   # Test sets for unseen families
│   └── README.md              # Dataset documentation and usage
│
├── Notebook/                  # Training and evaluation notebooks
│   ├── ModernBERT_base_DGA_wl_8F.ipynb         # ModernBERT Expert training ⭐
│   ├── ModernBERT_base_DGA_54F.ipynb           # ModernBERT Generalist
│   ├── Train_Gemma3_4B_DGA_WordList.ipynb      # Gemma 3 4B LoRA
│   ├── Train_llama3B_DGA_WordList.ipynb        # LLaMA 3.2 3B LoRA
│   ├── DomUrlBert.ipynb                        # DomBertUrl
│   ├── CNN_Patron_WL.ipynb                     # CNN Wordlist
│   ├── FANCI.ipynb                             # FANCI Random Forest
│   ├── Labin_wl.ipynb                          # LABin
│   ├── Charbot.ipynb                           # Generate CharBot
│   └── README.md                               # Notebook usage guide
│
├── Result_csv/                # Aggregated performance metrics (7 CSV files)
│   ├── ModernBERT_DGA_WL_metrics_summary.csv   # ModernBERT Expert ⭐
│   ├── ModernBERT_DGA_WL_54F_metrics_summary.csv
│   ├── DomBertUrl_DGA_WL_metrics_summary.csv
│   ├── df_results_gemma3_8B_WL.csv
│   ├── Llama3_8bits_metrics_summary.csv
│   ├── CNN_DGA_WL_metrics_summary.csv
│   ├── RF_DGA_WL_metrics_summary.csv
│   └── README.md              # Metrics documentation and usage examples
│
├── Result_File/               # Detailed per-batch results (2,640 files, 14MB)
│   ├── results_ModernBert_wl/
│   ├── results_ModernBert_wl_54families/
│   ├── results_dombert_url/
│   ├── results_gemma3_8B_wl/
│   ├── results_llama3_8bits/
│   ├── results_CNN_wl/
│   ├── results_FANCI_wl/
│   ├── results_Labin_wl/
│   └── README.md              # Detailed results documentation
│
├── LATAM_DGA_Detector-33.pdf  # Research paper (under review)
└── README.md                  # This file
```

---

## 🏆 Model Performance Comparison

### Overall Results

| Model | Known F1 | Unknown F1 | Inference | Throughput | GPU Required |
|-------|----------|------------|-----------|------------|--------------|
| **ModernBERT Expert** ⭐ | **86.7%** | **80.9%** | **26ms** | **38k/s** | Yes (Tesla T4) |
| ModernBERT Generalist | 79.2% | 62.1% | 27ms | 37k/s | Yes |
| DomBertUrl | 81.2% | **84.6%** | 28ms | 36k/s | Yes |
| Gemma 3 4B LoRA | 78.6% | 73.2% | 650ms | 1.5k/s | Yes (16GB+) |
| LLaMA 3.2 3B LoRA | 81.4% | 74.8% | 680ms | 1.4k/s | Yes (16GB+) |
| CNN Wordlist | 78.9% | 72.1% | **15ms** | **66k/s** | Yes (low mem) |
| FANCI (Random Forest) | 77.3% | 68.5% | **<1ms** | **>100k/s** | No (CPU only) |

### Key Observations

1. **ModernBERT Expert** achieves best balance of accuracy (86.7%), generalization (80.9%), and speed (26ms)
2. **DomBertUrl** shows exceptional generalization (84.6%) to unseen families
3. **Specialist training** (8 families) outperforms generalist (54 families) by +9.4% F1
4. **LLMs** (Gemma, LLaMA) achieve good accuracy but impractical inference times for real-time systems
5. **FANCI** offers fastest inference (<1ms CPU) but lower accuracy

---

## 📊 Evaluation Methodology

### Two-Phase Protocol

#### Phase 1: Known Families (In-Distribution)

- **Families**: charbot, deception, gozi, manuelita, matsnu, nymaim, rovnix, suppobox
- **Batches**: 30 random samples per family
- **Batch Size**: 100 DGA + 100 benign = 200 domains
- **Purpose**: Measure performance on training distribution

#### Phase 2: Generalization Test (Zero-Shot)

- **Families**: bigviktor, ngioweb, pizd
- **Purpose**: Test generalization to unseen wordlist-based DGAs
- **Evaluation**: Full datasets (no sampling)

### Metrics

- **F1-Score**: Primary metric (harmonic mean of precision/recall)
- **Precision**: DGA class precision (low false positives critical)
- **Recall**: DGA class recall (detection rate)
- **False Positive Rate (FPR)**: Benign domains misclassified as DGA
- **Inference Time**: Per-domain prediction time (ms)
- **Statistical Validation**: Mean ± std across 30 batches

---

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch transformers scikit-learn pandas numpy jupyter

# For GPU acceleration (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Optional: For specific models
pip install peft bitsandbytes  # LoRA adapters
pip install tensorflow         # LABin model
```

### Loading the Optimal Model (ModernBERT Expert)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load from HuggingFace
model = AutoModelForSequenceClassification.from_pretrained(
    "Reynier/moe-wordlist-dga-models",
    subfolder="models/modernbert-wordlist-expert"
)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# Inference
domain = "suppocratewayregion.com"
inputs = tokenizer(domain, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.softmax(outputs.logits, dim=-1)

# 0 = benign, 1 = DGA
print(f"DGA probability: {prediction[0][1]:.4f}")
```

### Training Your Own Expert

```bash
# Open the training notebook
jupyter notebook Notebook/ModernBERT_base_DGA_wl_8F.ipynb

# Or train from command line
python train_expert.py \
    --model answerdotai/ModernBERT-base \
    --train_data Dataset/train/train_wl.csv \
    --epochs 3 \
    --batch_size 16
```

### Reproducing Evaluation Results

```python
import pandas as pd

# Load aggregated metrics
df = pd.read_csv('Result_csv/ModernBERT_DGA_WL_metrics_summary.csv')

# View per-family performance
print(df[['family', 'f1_mean', 'f1_std', 'precision_mean', 'recall_mean']])

# Calculate overall F1-score (macro average)
overall_f1 = df['f1_mean'].mean()
print(f"Overall F1-Score: {overall_f1:.4f}")
```

---

## 📚 Datasets

### Training Dataset

- **File**: `Dataset/train/train_wl.csv`
- **Size**: 160,000 domains (balanced)
- **Families**: 8 wordlist-based DGA families + benign domains
- **Format**: `domain,label,family`

### Test Datasets

- **Known Families**: `Dataset/test-known/` (8 families, 30 batches each)
- **Generalization**: `Dataset/test-generalization/` (3 unseen families)

**Full dataset documentation**: See [`Dataset/README.md`](Dataset/README.md)

---

## 📈 Results

### Aggregated Metrics

All performance metrics (F1, precision, recall, FPR, inference time) aggregated across 30 evaluation batches are available in `Result_csv/`:

- **Per-family metrics**: Mean ± std for each DGA family
- **Usage examples**: Python code for loading and analyzing results
- **Statistical significance**: 95% confidence intervals

**Documentation**: See [`Result_csv/README.md`](Result_csv/README.md)

### Detailed Results

Domain-level predictions with confidence scores for all 2,640 evaluation batches (30 batches × 11 families × 8 models) are available in `Result_File/`:

- **Per-domain predictions**: true_label, predicted_label, confidence_score
- **Error analysis**: Identify false positives/negatives
- **Reproducibility**: Verify aggregated metrics

**Documentation**: See [`Result_File/README.md`](Result_File/README.md)

---

## 🤗 HuggingFace Repository

All trained models and datasets are hosted on HuggingFace for easy access:

**Repository**: [Reynier/moe-wordlist-dga-models](https://huggingface.co/Reynier/moe-wordlist-dga-models)

### Available Models

1. **ModernBERT Expert** (optimal) - `models/modernbert-wordlist-expert/`
2. **ModernBERT Generalist** - `models/modernbert-generalist-54f/`
3. **Gemma 3 4B LoRA** - `models/gemma-3-4b-lora/`
4. **LLaMA 3.2 3B LoRA** - `models/llama-3.2-3b-lora/`
5. **DomBertUrl** - `models/dombert-url/`
6. **CNN Wordlist** - `models/cnn-wordlist/`
7. **FANCI** - `models/fanci/`
8. **LABin** - `models/labin/`

**Model documentation**: See [`Models/README.md`](Models/README.md)

---

## 🔬 Research Highlights

### Specialist vs. Generalist Training

Comparing **ModernBERT Expert** (8 families) vs. **ModernBERT Generalist** (54 families):

| Metric | Expert | Generalist | Improvement |
|--------|--------|------------|-------------|
| F1 (Known) | 86.7% | 79.2% | **+9.4%** |
| F1 (Unknown) | 80.9% | 62.1% | **+30.2%** |
| Training Time | 2-3 hours | 8-10 hours | - |

**Key Finding**: Domain-specific expert training significantly improves both in-distribution performance and zero-shot generalization.

### Generalization Capability

Best performing models on **unseen families**:

1. **DomBertUrl**: 84.6% F1 (domain-pretrained advantage)
2. **ModernBERT Expert**: 80.9% F1 (optimal overall)
3. **LLaMA 3.2 3B**: 74.8% F1 (LLM generalization)

**Key Finding**: Domain-pretrained models (DomBertUrl) generalize better to unseen DGA families than general-purpose transformers.

### Inference Speed vs. Accuracy

Trade-off analysis for real-time deployment:

- **High Throughput (>30k/s)**: ModernBERT (38k/s), CNN (66k/s)
- **Ultra-Fast (<1ms)**: FANCI (CPU-only, lower accuracy)
- **High Accuracy**: ModernBERT (86.7%), LLaMA (81.4%)

**Key Finding**: ModernBERT offers optimal balance for production systems requiring both accuracy and real-time performance.

---

## 💡 Use Cases

### Production Deployment

```python
# Real-time DNS monitoring (38k domains/second)
from models import ModernBERTExpert

detector = ModernBERTExpert()
suspicious_domains = ["suppocratewayregion.com", "stableorderworldreign.net"]

for domain in suspicious_domains:
    score, label = detector.predict(domain)
    if label == "DGA":
        alert_security_team(domain, confidence=score)
```

### Threat Intelligence

```python
# Family attribution for incident response
detector = ModernBERTExpert(return_family=True)
domain = "maliciousword.biz"

prediction = detector.predict(domain)
print(f"Family: {prediction['family']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Research & Benchmarking

```python
# Evaluate new models against our baselines
from evaluation import TwoPhaseEvaluator

evaluator = TwoPhaseEvaluator(
    test_known='Dataset/test-known/',
    test_generalization='Dataset/test-generalization/'
)

results = evaluator.evaluate(your_model)
print(f"Known F1: {results['known_f1']:.4f}")
print(f"Unknown F1: {results['unknown_f1']:.4f}")
```

---

## 📝 Citation

This work is currently **under review**. Pre-print available in this repository ([LATAM_DGA_Detector-33.pdf](LATAM_DGA_Detector-33.pdf)).

If you use this code, models, or datasets in your research, please cite:

```bibtex
@article{leyva2025expert,
  title={Expert Selection for Wordlist-Based DGA Detection: A Systematic Evaluation},
  author={Leyva La O, Reynier and Catania, Carlos A. and Gonzalez, Rodrigo},
  journal={Under Review},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- **New Expert Models**: Implement and evaluate additional architectures
- **Extended Evaluation**: Test on additional DGA families
- **MoE Integration**: Complete router implementation for full MoE system
- **Optimization**: Improve inference speed and memory efficiency
- **Deployment**: Production-ready wrappers and APIs

---

## 📧 Contact

- **Author**: Reynier Leyva La O
- **Email**: rleyvalao@mendoza-conicet.gob.ar
- **HuggingFace**: [Reynier](https://huggingface.co/Reynier)
- **GitHub**: [reypapin](https://github.com/reypapin)

---

## 📄 License

This project is released for **research and educational purposes**. Please cite appropriately if used in academic work.

---

## 🙏 Acknowledgments

- **CONICET** (Consejo Nacional de Investigaciones Científicas y Técnicas)
- **Universidad Nacional de Cuyo**
- **HuggingFace** for model hosting and community support
- **DGA Research Community** for datasets and baseline implementations

---

**Last Updated**: October 2025
