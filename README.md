# Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach

This repository contains the implementation and comprehensive evaluation of candidate expert models for detecting wordlist-based Domain Generation Algorithms (DGAs). This research addresses expert model selection for wordlist-based DGA detection within a Mixture of Experts (MoE) architecture, focusing on models that balance detection accuracy, generalization capability, and computational efficiency for real-time cybersecurity systems.

## Abstract

Domain Generation Algorithms (DGAs) have evolved beyond traditional pseudorandom patterns, with wordlist-based variants generating linguistically coherent domains that evade conventional detection methods. While previous research has primarily focused on generalist detection approaches across multiple DGA types, systematic expert model selection specifically targeting wordlist-based variants remains largely unexplored. 

This work addresses expert model selection for wordlist-based DGA detection, where expert models refer to specialized architectures trained exclusively on specific DGA categories. We conduct systematic evaluation of seven candidate models across transformer, convolutional neural network (CNN), and traditional machine learning approaches. Models were trained on a balanced dataset of 160,000 domains spanning eight wordlist-based DGA families and evaluated using a rigorous two-phase protocol that measures both performance on training families and generalization to previously unseen variants. 

Our comparative analysis identifies fine-tuned **ModernBERT as the optimal expert model**, achieving **86.7% F1-score** on known families while maintaining **80.9% performance** on unknown families with **26ms inference time** on NVIDIA Tesla T4 GPUs, enabling processing of approximately **38,000 domains per second**. The study validates that domain-specific expert training significantly outperforms generalist approaches trained on diverse DGA families, with F1-score improvements of **9.4% on familiar variants** and **30.2% on unseen families**.

## 📁 Repository Structure

```
MoE-word-list-dga-detection/
├── Models/                          # Expert model candidates and configurations
│   ├── DomBertUrl/                 # Domain-specific BERT variant
│   ├── modernbert-dga-detector/    # Base ModernBERT expert (optimal model)
│   ├── modernbert-dga-detector-16families/  # ModernBERT trained on 16 families
│   ├── modernbert-dga-detector-46familias/  # ModernBERT trained on 46 families
│   ├── gemma_dga_detector/         # Gemma LLM fine-tuned expert
│   ├── gemma_2epoch_dector_dga/    # Gemma 2-epoch training variant
│   ├── llama3.2_3B_dectector_dga/  # Llama 3.2 3B expert model
│   ├── FANCI_model/                # FANCI traditional ML classifier
│   ├── LABIN/                      # LABin neural network approach
│   ├── dga_RF/                     # Random Forest baseline model
│   ├── dga_cnn_model_wl/           # CNN with wordlist features
│   └── README.md                   # Detailed model descriptions
│
├── Notebook/                        # Training and evaluation notebooks
│   ├── ModernBERT_base_DGA_Word.ipynb     # ModernBERT training pipeline
│   ├── ModernBERT_base_DGA_wl_8F.ipynb    # 8-family ModernBERT training
│   ├── Train_Gemma3_4B_DGA_WordList.ipynb # Gemma expert training
│   ├── Train_3B_DGA_WordList.ipynb        # 3B model training pipeline
│   ├── Test_Gemma3_4B_DGA_Last.ipynb      # Gemma evaluation
│   ├── Test__llama3B_DGA.ipynb            # Llama evaluation
│   ├── DomUrlBert.ipynb                   # Domain-URL BERT experiments
│   ├── RF_WL.ipynb                        # Random Forest baseline
│   ├── CNN_Patron_WL.ipynb               # CNN pattern recognition
│   ├── FANCI.ipynb                        # FANCI classifier experiments
│   ├── Labin_wl.ipynb                     # LABin model training
│   └── README.md                          # Notebook descriptions and usage
│
├── Result_File/                     # Detailed evaluation results per DGA family
│   ├── result_wl_Llama3B_8bits/    # Llama 3B 8-bit quantization results
│   ├── results_FANCI_wl/           # FANCI classifier detailed results
│   └── README.md                   # Results structure and analysis guide
│
├── Result_csv/                      # Performance metrics summaries
│   ├── CNN_DGA_WL_metrics_summary.csv        # CNN performance metrics
│   ├── DomBertUrl_DGA_WL_metrics_summary.csv # DomBERT evaluation results
│   ├── Llama3_8bits_metrics_summary.csv      # Llama quantized results
│   ├── ModernBERT_DGA_WL_metrics_summary.csv # ModernBERT expert metrics
│   ├── RF_DGA_WL_metrics_summary.csv         # Random Forest baseline
│   ├── df_results_gemma3_8B_WL.csv           # Gemma detailed results
│   └── README.md                             # Metrics explanation and usage
│
├── Paper/                           # Research papers and references
│   ├── Latam/                      # LATAM cybersecurity research
│   │   ├── CACIC2025/             # CACIC 2025 conference submission
│   │   └── [Various DGA papers]   # State-of-the-art references
│   └── README.md                   # Paper organization and citations
│
├── train_wl.csv                    # Training dataset (160,000 domains, 8 families)
├── LARGE_FILES.md                  # Instructions for accessing large model files
└── README.md                       # This file
```

## 🎯 Research Contributions

### 1. Systematic Evaluation Framework
Comprehensive methodology for evaluating candidate expert models in MoE architectures specifically targeting wordlist-based DGA detection, incorporating both performance and operational constraints.

### 2. Comprehensive Empirical Analysis
Rigorous evaluation of seven state-of-the-art models across multiple DGA families, including explicit out-of-family generalization testing to assess robustness against emerging threats.

### 3. Composite Performance Metric
Novel evaluation metric integrating detection accuracy (precision, recall, F1-score), operational reliability (false positive rate), and deployment feasibility (inference time) for holistic model comparison.

### 4. Performance Characterization
Identification of **ModernBERT as the optimal expert** for wordlist-based detection, achieving superior performance with practical deployment characteristics.

### 5. Practical Deployment Guidelines
Actionable recommendations for expert selection in production MoE systems, including characterization of model strengths and limitations across different DGA variants.

## 🚀 Key Findings

### Optimal Expert Model: ModernBERT
- **F1-Score**: 86.7% on known families, 80.9% on unknown families
- **Inference Time**: 26ms on NVIDIA Tesla T4 GPU
- **Throughput**: ~38,000 domains/second
- **Improvement**: 9.4% better than generalist approaches on known families, 30.2% on unseen families

### Model Performance Comparison
| Model | Known Families F1 | Unknown Families F1 | Inference Time | Throughput |
|-------|------------------|---------------------|----------------|------------|
| ModernBERT | **86.7%** | **80.9%** | **26ms** | **38k/s** |
| Gemma 3B | 82.1% | 75.3% | 650ms | 1.5k/s |
| Llama 3.2 3B | 81.4% | 74.8% | 680ms | 1.4k/s |
| CNN | 78.9% | 72.1% | 15ms | 66k/s |

## 📊 Evaluation Protocol

### Two-Phase Evaluation
1. **Phase 1**: Performance on training families (known variants)
2. **Phase 2**: Generalization to unseen DGA families (unknown variants)

### DGA Families Evaluated
**Training Families (8)**: bigviktor, charbot, deception, gozi, manuelita, matsnu, ngioweb, nymaim, pizd, rovnix, suppobox

**Testing Families**: Additional wordlist-based variants for generalization assessment

### Metrics
- **Detection Accuracy**: Precision, Recall, F1-Score
- **Operational Reliability**: False Positive Rate, Specificity
- **Deployment Feasibility**: Inference Time, Memory Usage
- **Generalization**: Performance on unseen families

## 💻 Getting Started

### Prerequisites
```bash
pip install torch transformers sklearn pandas numpy jupyter
# For GPU acceleration
pip install torch[cuda] # or appropriate CUDA version
```

### Quick Start
1. **Load Pre-trained Expert**: Use ModernBERT from `Models/modernbert-dga-detector/`
2. **Training**: Follow notebooks in `Notebook/` for retraining
3. **Evaluation**: Check `Result_csv/` for performance metrics
4. **Inference**: See model-specific READMEs for usage examples

### Dataset Format
The training dataset (`train_wl.csv`) contains:
- Domain names with wordlist-based features
- DGA family labels for 8 wordlist-based families
- Balanced distribution (160,000 total domains)

## 🏆 Impact and Applications

### Cybersecurity Applications
- **Real-time DNS Monitoring**: 38k domains/second processing capability
- **Threat Intelligence**: Family-specific attribution and analysis
- **Incident Response**: Rapid classification of suspicious domains
- **Network Security**: Integration with existing security infrastructure

### Research Impact
- **Reproducible Evaluation**: Complete codebase and dataset availability
- **Benchmark Establishment**: Standard evaluation protocol for wordlist-based DGA detection
- **Expert Selection Guidelines**: Systematic methodology for MoE component selection

## 📚 Citation

This work is currently under review and will be updated with citation information once published.

## 🤝 Contributing

This research project welcomes contributions in the following areas:
- Additional expert model implementations
- Extended evaluation on new DGA families
- Integration with complete MoE architectures
- Performance optimizations for production deployment

## 📄 License

This project is released for research purposes. Please cite appropriately if used in academic work.

---

**Note**: Due to GitHub file size limitations, large model weights (>100MB) are documented in `LARGE_FILES.md` with instructions for access. All configuration files, tokenizers, and smaller models are included in the repository.