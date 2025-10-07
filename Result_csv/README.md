# Performance Metrics Summary

This directory contains aggregated performance metrics for all seven expert models evaluated on wordlist-based DGA detection.

---

## 📂 Files

| File | Model | Size |
|------|-------|------|
| `ModernBERT_DGA_WL_metrics_summary.csv` | ModernBERT Expert (Optimal) | 3.2 KB |
| `ModernBERT_DGA_WL_54F_metrics_summary.csv` | ModernBERT Generalist | 3.1 KB |
| `DomBertUrl_DGA_WL_metrics_summary.csv` | DomBertUrl | 3.1 KB |
| `df_results_gemma3_8B_WL.csv` | Gemma 3 4B LoRA | 3.1 KB |
| `Llama3_8bits_metrics_summary.csv` | LLaMA 3.2 3B LoRA | 3.2 KB |
| `CNN_DGA_WL_metrics_summary.csv` | CNN Wordlist | 3.3 KB |
| `RF_DGA_WL_metrics_summary.csv` | FANCI Random Forest | 3.0 KB |

**Total Size:** ~22 KB

---

## 📊 CSV Format

Each file contains performance metrics **per DGA family** across **30 evaluation batches**:

### Columns

```csv
family,accuracy_mean,accuracy_std,f1_mean,f1_std,precision_mean,precision_std,recall_mean,recall_std,fpr_mean,fpr_std,tpr_mean,tpr_std,query_time_mean,query_time_std,total_runs
```

### Column Descriptions

| Column | Description | Unit |
|--------|-------------|------|
| `family` | DGA family name or "legit" for benign | string |
| `accuracy_mean` | Mean accuracy across 30 batches | 0.0-1.0 |
| `accuracy_std` | Standard deviation of accuracy | 0.0-1.0 |
| `f1_mean` | Mean F1-score (primary metric) | 0.0-1.0 |
| `f1_std` | Standard deviation of F1-score | 0.0-1.0 |
| `precision_mean` | Mean precision (DGA class) | 0.0-1.0 |
| `precision_std` | Standard deviation of precision | 0.0-1.0 |
| `recall_mean` | Mean recall/sensitivity (DGA class) | 0.0-1.0 |
| `recall_std` | Standard deviation of recall | 0.0-1.0 |
| `fpr_mean` | Mean False Positive Rate | 0.0-1.0 |
| `fpr_std` | Standard deviation of FPR | 0.0-1.0 |
| `tpr_mean` | Mean True Positive Rate (=recall) | 0.0-1.0 |
| `tpr_std` | Standard deviation of TPR | 0.0-1.0 |
| `query_time_mean` | Mean inference time per domain | seconds |
| `query_time_std` | Standard deviation of inference time | seconds |
| `total_runs` | Number of evaluation batches | integer (30) |

---

## 🎯 Evaluation Protocol

### Known Families (8 families)
- **Families:** charbot, deception, gozi, manuelita, matsnu, nymaim, rovnix, suppobox
- **Batches:** 30 random samples per family
- **Batch Size:** 100 DGA domains + 100 benign domains = 200 total
- **Purpose:** Measure performance on training distribution

### Generalization Test (3 families)
- **Families:** bigviktor, ngioweb, pizd
- **Purpose:** Test generalization to unseen wordlist-based DGAs
- **Note:** Full datasets tested (no random sampling)

---

## 📈 Quick Performance Summary

### ModernBERT Expert (Optimal) ⭐

**Known Families:**
- **F1-Score:** 86.7% ± 3.0%
- **Precision:** 89.7% ± 4.1%
- **Recall:** 86.6% ± 3.1%
- **FPR:** 9.0% ± 3.8%
- **Inference:** 26ms per domain

**Unknown Families:**
- **F1-Score:** 80.9% ± 4.5%
- **Generalization Gap:** ~6%

### Model Comparison

| Model | Known F1 | Unknown F1 | Inference |
|-------|----------|------------|-----------|
| **ModernBERT Expert** ⭐ | **86.7%** | **80.9%** | **26ms** |
| ModernBERT Generalist | 79.2% | 62.1% | 27ms |
| DomBertUrl | 81.2% | **84.6%** | 28ms |
| Gemma 3 4B | 78.6% | 73.2% | 650ms |
| LLaMA 3.2 3B | 81.4% | 74.8% | 680ms |
| CNN | 78.9% | 72.1% | **15ms** |
| FANCI (RF) | 77.3% | 68.5% | **<1ms** |

---

## 💻 Usage Examples

### Load and View Results

```python
import pandas as pd

# Load ModernBERT results
df = pd.read_csv('ModernBERT_DGA_WL_metrics_summary.csv')

# View per-family performance
print(df[['family', 'f1_mean', 'f1_std', 'query_time_mean']])

# Calculate overall F1-score (macro average)
overall_f1 = df['f1_mean'].mean()
print(f"Overall F1-Score: {overall_f1:.4f}")
```

### Compare Multiple Models

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all models
models = {
    'ModernBERT': pd.read_csv('ModernBERT_DGA_WL_metrics_summary.csv'),
    'DomBertUrl': pd.read_csv('DomBertUrl_DGA_WL_metrics_summary.csv'),
    'CNN': pd.read_csv('CNN_DGA_WL_metrics_summary.csv'),
    'RF': pd.read_csv('RF_DGA_WL_metrics_summary.csv'),
}

# Compare F1-scores
comparison = {}
for name, df in models.items():
    comparison[name] = df['f1_mean'].mean()

# Plot comparison
plt.bar(comparison.keys(), comparison.values())
plt.ylabel('Mean F1-Score')
plt.title('Model Comparison: Overall F1-Score')
plt.ylim([0, 1])
plt.show()
```

### Extract Per-Family Performance

```python
import pandas as pd

df = pd.read_csv('ModernBERT_DGA_WL_metrics_summary.csv')

# Get best performing family
best_family = df.loc[df['f1_mean'].idxmax()]
print(f"Best Family: {best_family['family']}")
print(f"F1-Score: {best_family['f1_mean']:.4f} ± {best_family['f1_std']:.4f}")

# Get worst performing family
worst_family = df.loc[df['f1_mean'].idxmin()]
print(f"Worst Family: {worst_family['family']}")
print(f"F1-Score: {worst_family['f1_mean']:.4f} ± {worst_family['f1_std']:.4f}")

# Calculate per-family variance
df['coefficient_of_variation'] = df['f1_std'] / df['f1_mean']
most_stable = df.loc[df['coefficient_of_variation'].idxmin()]
print(f"Most Stable: {most_stable['family']} (CV={most_stable['coefficient_of_variation']:.4f})")
```

### Generate Performance Report

```python
import pandas as pd

def generate_report(csv_file, model_name):
    df = pd.read_csv(csv_file)

    print(f"=== {model_name} Performance Report ===")
    print(f"\nOverall Metrics (Mean ± Std):")
    print(f"  F1-Score:  {df['f1_mean'].mean():.4f} ± {df['f1_std'].mean():.4f}")
    print(f"  Precision: {df['precision_mean'].mean():.4f} ± {df['precision_std'].mean():.4f}")
    print(f"  Recall:    {df['recall_mean'].mean():.4f} ± {df['recall_std'].mean():.4f}")
    print(f"  FPR:       {df['fpr_mean'].mean():.4f} ± {df['fpr_std'].mean():.4f}")
    print(f"  Inference: {df['query_time_mean'].mean()*1000:.2f}ms ± {df['query_time_std'].mean()*1000:.2f}ms")

    print(f"\nPer-Family F1-Scores:")
    for _, row in df.iterrows():
        print(f"  {row['family']:12s}: {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")

# Example usage
generate_report('ModernBERT_DGA_WL_metrics_summary.csv', 'ModernBERT Expert')
```

---

## 📊 Statistical Significance

All metrics are computed across **30 independent batches** with:
- **Mean:** Central tendency measure
- **Standard Deviation:** Variability across batches
- **Confidence Interval (95%):** mean ± 1.96 × std

### Example Interpretation

```
F1-Score: 0.867 ± 0.030
```

- **Mean:** 86.7%
- **95% CI:** [80.8%, 92.6%]
- **Interpretation:** Model achieves 86.7% F1-score with high consistency across batches

---

## 🔍 Key Findings

### 1. Specialist vs. Generalist

Comparing `ModernBERT_DGA_WL_metrics_summary.csv` vs `ModernBERT_DGA_WL_54F_metrics_summary.csv`:

- **Specialist (8 families):** 86.7% F1 (known), 80.9% F1 (unknown)
- **Generalist (54 families):** 79.2% F1 (known), 62.1% F1 (unknown)
- **Improvement:** +9.4% (known), +30.2% (unknown)

**Conclusion:** Domain-specific training significantly improves performance.

### 2. Generalization Capability

Best generalization (unknown families):
1. **DomBertUrl:** 84.6% F1 (best)
2. **ModernBERT Expert:** 80.9% F1
3. **LLaMA 3.2 3B:** 74.8% F1

**Conclusion:** Domain-pretrained models generalize better to unseen DGA families.

### 3. Inference Speed

Fastest models:
1. **FANCI (RF):** <1ms (CPU-only)
2. **CNN:** 15ms (GPU)
3. **ModernBERT:** 26ms (GPU)

**Conclusion:** ModernBERT offers optimal balance of accuracy and speed for real-time deployment.

---

## 📖 Related Files

- **Detailed Results:** [`../Result_File/`](../Result_File/) - Per-batch predictions
- **Training Notebooks:** [`../Notebook/`](../Notebook/) - Model training scripts
- **Datasets:** [`../Dataset/`](../Dataset/) - Training and test data
- **Models:** Available on [HuggingFace](https://huggingface.co/Reynier/moe-wordlist-dga-models)

---

## 📞 Citation

If you use these results in your research, please cite:

```bibtex
@article{leyva2025expert,
  title={Expert Selection for Wordlist-Based DGA Detection: A Systematic Evaluation},
  author={Leyva La O, Reynier and Catania, Carlos A. and Gonzalez, Rodrigo},
  journal={Under Review},
  year={2025}
}
```

---

**Last Updated:** October 2025
