# Notebooks Directory

This directory contains Jupyter notebooks for training, evaluation, and experimentation with different DGA detection models.

## Training Notebooks

### BERT-based Models
- **ModernBERT_base_DGA_Word.ipynb**: Training ModernBERT for word-based DGA detection
- **ModernBERT_base_DGA_wl_8F.ipynb**: ModernBERT training with 8 DGA families
- **DomUrlBert.ipynb**: Domain-URL BERT model training

### Large Language Models
- **Train_Gemma3_4B_DGA_WordList.ipynb**: Gemma 3B/4B training pipeline
- **Train_3B_DGA_WordList.ipynb**: General 3B model training
- **Test_Gemma3_4B_DGA_Last.ipynb**: Final Gemma model testing
- **Test__Gemma3_4B_DGA.ipynb**: Gemma model evaluation
- **Test__llama3B_DGA.ipynb**: Llama 3B model testing

### Traditional & Deep Learning Models
- **RF_WL.ipynb**: Random Forest training with word-list features
- **CNN_Patron_WL.ipynb**: CNN pattern recognition for word-list based detection
- **FANCI.ipynb**: FANCI classifier implementation
- **Labin_wl.ipynb**: LABin model training with word-list features

## Notebook Categories

### Training Notebooks
- Complete training pipelines for each model type
- Data preprocessing and feature engineering
- Model architecture definition
- Training loop implementation

### Testing/Evaluation Notebooks
- Model performance evaluation
- Cross-validation experiments
- Comparison studies between different approaches
- Results visualization and analysis

## Usage Instructions

1. **Environment Setup**: Ensure all required packages are installed
2. **Data Preparation**: Place training data in the appropriate directory
3. **Model Training**: Run training notebooks sequentially
4. **Evaluation**: Use testing notebooks for model assessment

## Key Features

- Comprehensive model comparison framework
- Word-list based feature extraction
- Multi-family DGA detection capabilities
- Performance metrics calculation
- Results visualization

## Requirements

```
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```