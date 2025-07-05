# Results Summary Directory

This directory contains aggregated performance metrics and summary statistics for all evaluated DGA detection models.

## Summary Files

### Model Performance Summaries

#### **CNN_DGA_WL_metrics_summary.csv**
Performance metrics for the Convolutional Neural Network model with word-list features:
- Architecture: CNN with pattern recognition layers
- Features: Word-list based domain characteristics
- Evaluation: Cross-family validation results

#### **DomBertUrl_DGA_WL_metrics_summary.csv**
Performance metrics for the Domain-URL BERT model:
- Architecture: Fine-tuned BERT for domain classification
- Features: Domain and URL linguistic patterns
- Specialization: Domain-specific language understanding

#### **Llama3_8bits_metrics_summary.csv**
Comprehensive evaluation results for Llama 3.2 3B model with 8-bit quantization:
- Architecture: Large Language Model with fine-tuning
- Optimization: 8-bit quantization for efficiency
- Coverage: Extensive multi-family evaluation

#### **ModernBERT_DGA_WL_metrics_summary.csv**
Performance metrics for ModernBERT variants:
- Architecture: State-of-the-art BERT improvements
- Variants: Base, 16-families, 46-families models
- Features: Advanced transformer-based classification

#### **RF_DGA_WL_metrics_summary.csv**
Random Forest classifier performance with word-list features:
- Architecture: Ensemble decision trees
- Features: Engineered word-list characteristics
- Baseline: Traditional ML approach for comparison

#### **df_results_gemma3_8B_WL.csv**
Detailed results for Gemma 3B/8B models:
- Architecture: Google's Gemma language models
- Scale: Both 3B and 8B parameter variants
- Approach: Fine-tuning for DGA classification

## Metrics Included

Each summary file typically contains:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged precision
- **Recall**: Per-class and macro-averaged recall  
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **NPV**: Negative predictive value

### Performance Analysis
- **Per-Family Results**: Individual DGA family performance
- **Macro Averages**: Unweighted averages across families
- **Weighted Averages**: Sample-weighted performance metrics
- **Confusion Matrices**: Detailed classification breakdowns

### Statistical Measures
- **Standard Deviation**: Performance variance across folds
- **Confidence Intervals**: Statistical significance bounds
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve

## Usage Examples

### Loading Summary Data
```python
import pandas as pd

# Load CNN results
cnn_results = pd.read_csv('CNN_DGA_WL_metrics_summary.csv')

# Load Llama results
llama_results = pd.read_csv('Llama3_8bits_metrics_summary.csv')

# Compare model performance
models = ['CNN', 'ModernBERT', 'Llama3', 'RF']
accuracies = [cnn_results['accuracy'].mean(), 
              modernbert_results['accuracy'].mean(),
              llama_results['accuracy'].mean(),
              rf_results['accuracy'].mean()]
```

### Performance Comparison
```python
# Create comparative analysis
import matplotlib.pyplot as plt

# Load all model summaries
model_files = {
    'CNN': 'CNN_DGA_WL_metrics_summary.csv',
    'DomBERT': 'DomBertUrl_DGA_WL_metrics_summary.csv',
    'Llama3': 'Llama3_8bits_metrics_summary.csv',
    'ModernBERT': 'ModernBERT_DGA_WL_metrics_summary.csv',
    'Random Forest': 'RF_DGA_WL_metrics_summary.csv',
    'Gemma': 'df_results_gemma3_8B_WL.csv'
}

# Generate comparison plots
for model, file in model_files.items():
    data = pd.read_csv(file)
    plt.bar(model, data['f1_score'].mean())
plt.title('Model Comparison: F1-Score')
plt.show()
```

## Key Insights

### Model Rankings
Based on aggregated performance across DGA families:
1. **Large Language Models** (Llama, Gemma): Superior contextual understanding
2. **Modern Transformers** (ModernBERT): Excellent balance of performance and efficiency
3. **Domain-Specific Models** (DomBERT): Strong domain-focused performance
4. **Deep Learning** (CNN): Good pattern recognition capabilities
5. **Traditional ML** (Random Forest): Reliable baseline performance

### Family-Specific Performance
- **High Accuracy Families**: Families with distinct linguistic patterns
- **Challenging Families**: Families with human-like domain patterns
- **Cross-Family Generalization**: Models' ability to detect unseen DGA families

### Computational Efficiency
- **Training Time**: Model training duration comparisons
- **Inference Speed**: Real-time classification capabilities
- **Memory Usage**: Resource requirements for deployment

## Recommendations

Based on the evaluation results:
- **Best Overall**: Llama/Gemma models for comprehensive accuracy
- **Best Efficiency**: ModernBERT for balanced performance/speed
- **Best Deployment**: CNN for resource-constrained environments
- **Best Baseline**: Random Forest for interpretable results