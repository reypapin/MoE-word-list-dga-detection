# Llama 3.2 3B 8-bit Quantization Results

This directory contains detailed evaluation results for the Llama 3.2 3B model with 8-bit quantization across multiple wordlist-based DGA families. The results demonstrate the performance characteristics of large language models in the context of expert model selection for DGA detection.

## Evaluation Overview

### Model Configuration
- **Model**: Llama 3.2 3B with 8-bit quantization
- **Quantization**: 8-bit precision for memory efficiency
- **Inference Hardware**: NVIDIA Tesla T4 GPU
- **Batch Processing**: Results split across 30 batches per family
- **Evaluation Period**: Comprehensive cross-family testing

### Performance Summary
- **Known Families F1**: ~81.4%
- **Unknown Families F1**: ~74.8%
- **Inference Time**: ~680ms per domain
- **Throughput**: ~1,400 domains/second
- **Memory Usage**: Reduced due to 8-bit quantization

## DGA Families Evaluated

### Tested Families (11 families)
Each family has 30 result files (batches 0-29):

#### 1. bigviktor
- **Files**: `results_Llama3_FineTuning_docker_bigviktor.gz_*.csv.gz`
- **Characteristics**: Large-scale botnet with extensive domain generation
- **Pattern Type**: Dictionary-based concatenation with technical terms

#### 2. charbot
- **Files**: `results_Llama3_FineTuning_docker_charbot.gz_*.csv.gz`
- **Characteristics**: Chat-based malware communication
- **Pattern Type**: Conversational and social media inspired domains

#### 3. deception
- **Files**: `results_Llama3_FineTuning_docker_deception.gz_*.csv.gz`
- **Characteristics**: Anti-analysis techniques
- **Pattern Type**: Deceptive domains mimicking legitimate services

#### 4. gozi
- **Files**: `results_Llama3_FineTuning_docker_gozi.gz_*.csv.gz`
- **Characteristics**: Banking trojan with sophisticated patterns
- **Pattern Type**: Financial and business-oriented wordlists

#### 5. manuelita
- **Files**: `results_Llama3_FineTuning_docker_manuelita.gz_*.csv.gz`
- **Characteristics**: Regional-specific malware
- **Pattern Type**: Linguistically sophisticated, human-like patterns

#### 6. matsnu
- **Files**: `results_Llama3_FineTuning_docker_matsnu.gz_*.csv.gz`
- **Characteristics**: Banking trojan variant
- **Pattern Type**: Technical and financial terminology

#### 7. ngioweb
- **Files**: `results_Llama3_FineTuning_docker_ngioweb.gz_*.csv.gz`
- **Characteristics**: Web-based attack vectors
- **Pattern Type**: Web service and portal naming patterns

#### 8. nymaim
- **Files**: `results_Llama3_FineTuning_docker_nymaim.gz_*.csv.gz`
- **Characteristics**: Multi-purpose malware platform
- **Pattern Type**: Diverse wordlist combinations

#### 9. pizd
- **Files**: `results_Llama3_FineTuning_docker_pizd.gz_*.csv.gz`
- **Characteristics**: Specialized malware family
- **Pattern Type**: Technical and system-oriented terms

#### 10. rovnix
- **Files**: `results_Llama3_FineTuning_docker_rovnix.gz_*.csv.gz`
- **Characteristics**: Financial malware with advanced evasion
- **Pattern Type**: Banking and financial service mimicry

#### 11. suppobox
- **Files**: `results_Llama3_FineTuning_docker_suppobox.gz_*.csv.gz`
- **Characteristics**: Support infrastructure malware
- **Pattern Type**: Technical support and service domains

## File Format and Structure

### File Naming Convention
```
results_Llama3_FineTuning_docker_{FAMILY_NAME}.gz_{BATCH_NUMBER}.csv.gz
```

### Content Structure
Each CSV file contains detailed per-domain evaluation results:

```csv
domain,true_label,predicted_label,confidence_score,family_predicted,timestamp
example-domain.com,1,1,0.892,gozi,2024-12-01T10:30:15
secure-portal.net,0,1,0.654,legitimate,2024-12-01T10:30:16
...
```

### Data Fields
- **domain**: The evaluated domain name
- **true_label**: Ground truth label (0=benign, 1=DGA)
- **predicted_label**: Model prediction (0=benign, 1=DGA)
- **confidence_score**: Prediction confidence (0-1)
- **family_predicted**: Predicted DGA family
- **timestamp**: Evaluation timestamp

## Evaluation Methodology

### Two-Phase Protocol
1. **Phase 1**: Performance on training families (in-distribution)
2. **Phase 2**: Generalization to unseen families (out-of-distribution)

### Batch Processing
- **Batch Size**: Variable based on family size
- **Total Batches**: 30 per family for statistical robustness
- **Processing**: Parallel evaluation for efficiency
- **Aggregation**: Results combined for family-level metrics

### Metrics Calculated
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate per class
- **Recall**: Sensitivity for DGA detection
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **ROC-AUC**: Area under receiver operating characteristic

## Performance Analysis

### Llama 3.2 3B Characteristics

#### Strengths
- **Semantic Understanding**: Strong linguistic pattern recognition
- **Generalization**: Reasonable performance on unseen families
- **Contextual Analysis**: Ability to understand domain semantics
- **Multi-family Detection**: Effective across diverse DGA types

#### Limitations
- **Inference Speed**: Significantly slower than specialized models
- **Resource Requirements**: High memory and computational needs
- **Deployment Complexity**: Requires GPU infrastructure
- **Quantization Effects**: 8-bit precision may impact accuracy

### Comparative Performance
- **vs. ModernBERT**: Lower accuracy but stronger semantic understanding
- **vs. Traditional ML**: Higher accuracy but much slower
- **vs. CNN**: Better generalization but significantly slower
- **vs. Other LLMs**: Balanced performance in LLM category

## Usage Instructions

### Extracting Results
```bash
# Extract a specific family's results
cd Result_File/result_wl_Llama3B_8bits/
gunzip results_Llama3_FineTuning_docker_gozi.gz_0.csv.gz

# View first few lines
head results_Llama3_FineTuning_docker_gozi.gz_0.csv
```

### Aggregating Family Results
```python
import pandas as pd
import glob
import gzip

def load_family_results(family_name, data_dir):
    """Load all batch results for a specific family"""
    pattern = f"results_Llama3_FineTuning_docker_{family_name}.gz_*.csv.gz"
    files = glob.glob(f"{data_dir}/{pattern}")
    
    dfs = []
    for file in sorted(files):
        with gzip.open(file, 'rt') as f:
            df = pd.read_csv(f)
            df['batch'] = int(file.split('_')[-1].split('.')[0])
            dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# Example usage
family_results = load_family_results('gozi', 'Result_File/result_wl_Llama3B_8bits/')
print(f"Total samples for gozi: {len(family_results)}")
print(f"Accuracy: {(family_results['true_label'] == family_results['predicted_label']).mean():.3f}")
```

### Performance Calculation
```python
from sklearn.metrics import classification_report, confusion_matrix

def calculate_family_metrics(family_results):
    """Calculate comprehensive metrics for family results"""
    y_true = family_results['true_label']
    y_pred = family_results['predicted_label']
    
    # Basic metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'confusion_matrix': cm
    }

# Example calculation
metrics = calculate_family_metrics(family_results)
print(f"F1-Score: {metrics['f1_score']:.3f}")
```

## Research Context

### Role in Expert Selection
These Llama 3.2 3B results serve multiple purposes in our research:

1. **LLM Baseline**: Performance benchmark for large language models
2. **Quantization Analysis**: Impact of 8-bit quantization on accuracy
3. **Speed-Accuracy Trade-off**: Comparison with faster alternatives
4. **Generalization Study**: Cross-family performance analysis

### Key Findings
- **Semantic Capability**: Strong semantic understanding of domain patterns
- **Deployment Challenges**: Inference speed limitations for real-time use
- **Quantization Impact**: Minimal accuracy loss with significant efficiency gains
- **Family Specificity**: Variable performance across different DGA families

## Citation

These results are part of our expert model selection research:

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={CACIC 2025},
  year={2024}
}
```