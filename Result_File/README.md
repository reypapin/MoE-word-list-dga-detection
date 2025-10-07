# Detailed Evaluation Results

This directory contains **detailed per-batch evaluation results** for all seven expert models across 11 DGA families. Each file contains domain-level predictions with confidence scores for comprehensive error analysis.

---

## 📂 Directory Structure

```
Result_File/
├── results_CNN_wl/                    # CNN Wordlist (330 files)
├── results_FANCI_wl/                  # FANCI Random Forest (330 files)
├── results_Labin_wl/                  # LABin (330 files)
├── results_ModernBert_wl/             # ModernBERT Expert ⭐ (330 files)
├── results_ModernBert_wl_54families/  # ModernBERT Generalist (330 files)
├── results_dombert_url/               # DomBertUrl (330 files)
├── results_gemma3_8B_wl/              # Gemma 3 4B LoRA (330 files)
└── results_llama3_8bits/              # LLaMA 3.2 3B LoRA (330 files)
```

**Total Files:** 2,640 compressed CSV files
**Total Size:** 14 MB (compressed)
**Files per Model:** 330 (30 batches × 11 families)

---

## 📋 File Naming Convention

```
results_{MODEL}_{FAMILY}.gz_{BATCH}.csv.gz
```

### Components

| Component | Description | Examples |
|-----------|-------------|----------|
| `MODEL` | Expert model name | `ModernBert_wl`, `FANCI_wl`, `gemma3_8B_wl` |
| `FAMILY` | DGA family or "legit" | `charbot`, `gozi`, `nymaim`, `legit` |
| `BATCH` | Batch number (0-29) | `0`, `15`, `29` |

### Examples

```
results_ModernBert_wl_charbot.gz_0.csv.gz       # ModernBERT, charbot family, batch 0
results_FANCI_wl_legit.gz_15.csv.gz             # FANCI, benign domains, batch 15
results_gemma3_8B_wl_bigviktor.gz_29.csv.gz     # Gemma, bigviktor family, batch 29
```

---

## 📊 File Format

Each compressed CSV file contains **domain-level predictions** with the following structure:

### Columns

```csv
domain,true_label,predicted_label,confidence_score
```

### Column Descriptions

| Column | Description | Type | Range |
|--------|-------------|------|-------|
| `domain` | Domain name evaluated | string | - |
| `true_label` | Ground truth label | integer | 0 (benign), 1 (DGA) |
| `predicted_label` | Model prediction | integer | 0 (benign), 1 (DGA) |
| `confidence_score` | Prediction confidence | float | 0.0-1.0 |

### Example Content

```csv
domain,true_label,predicted_label,confidence_score
suppocratewayregion.com,1,1,0.9876
stableorderworldreign.net,1,1,0.9543
google.com,0,0,0.9821
facebook.com,0,0,0.9765
maliciousword.biz,1,0,0.6234
```

---

## 🎯 Evaluation Structure

### Known Families (8 families)

**Families:** charbot, deception, gozi, manuelita, matsnu, nymaim, rovnix, suppobox

- **Batches per Family:** 30 random samples
- **Domains per Batch:** 100 DGA + 100 benign = 200 total
- **Files per Family:** 30
- **Purpose:** Measure in-distribution performance

**Example Files:**
```
results_ModernBert_wl_charbot.gz_0.csv.gz     # Batch 0: 200 domains
results_ModernBert_wl_charbot.gz_1.csv.gz     # Batch 1: 200 domains
...
results_ModernBert_wl_charbot.gz_29.csv.gz    # Batch 29: 200 domains
```

### Generalization Test (3 families)

**Families:** bigviktor, ngioweb, pizd

- **Purpose:** Test zero-shot generalization to unseen wordlist-based DGAs
- **Note:** Full datasets evaluated (no random sampling)
- **Files per Family:** 1 file (complete dataset)

**Example Files:**
```
results_ModernBert_wl_bigviktor.gz_0.csv.gz   # Complete bigviktor dataset
results_ModernBert_wl_ngioweb.gz_0.csv.gz     # Complete ngioweb dataset
results_ModernBert_wl_pizd.gz_0.csv.gz        # Complete pizd dataset
```

---

## 💻 Usage Examples

### Load Single Batch Result

```python
import gzip
import csv

def load_batch_results(model, family, batch):
    """Load detailed results for a specific batch"""
    file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        results = []
        for row in reader:
            results.append({
                'domain': row['domain'],
                'true_label': int(row['true_label']),
                'predicted_label': int(row['predicted_label']),
                'confidence_score': float(row['confidence_score'])
            })
    return results

# Example usage
results = load_batch_results('ModernBert_wl', 'charbot', 0)
print(f"Loaded {len(results)} predictions")
print(f"First domain: {results[0]}")
```

### Analyze Prediction Errors

```python
import gzip
import csv

def analyze_errors(model, family, batch):
    """Find all misclassified domains in a batch"""
    file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

    errors = {
        'false_positives': [],  # Benign predicted as DGA
        'false_negatives': []   # DGA predicted as benign
    }

    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = int(row['true_label'])
            predicted_label = int(row['predicted_label'])

            if true_label != predicted_label:
                error_info = {
                    'domain': row['domain'],
                    'confidence': float(row['confidence_score'])
                }

                if true_label == 0 and predicted_label == 1:
                    errors['false_positives'].append(error_info)
                elif true_label == 1 and predicted_label == 0:
                    errors['false_negatives'].append(error_info)

    return errors

# Example: Find all errors in batch 0
errors = analyze_errors('ModernBert_wl', 'charbot', 0)
print(f"False Positives: {len(errors['false_positives'])}")
print(f"False Negatives: {len(errors['false_negatives'])}")

# Show false negatives (missed DGAs)
if errors['false_negatives']:
    print("\nMissed DGA domains:")
    for error in errors['false_negatives'][:5]:
        print(f"  {error['domain']} (confidence: {error['confidence']:.4f})")
```

### Calculate Per-Batch Metrics

```python
import gzip
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_batch_metrics(model, family, batch):
    """Calculate performance metrics for a single batch"""
    file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

    y_true = []
    y_pred = []

    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row['true_label']))
            y_pred.append(int(row['predicted_label']))

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, pos_label=1),
        'precision': precision_score(y_true, y_pred, pos_label=1),
        'recall': recall_score(y_true, y_pred, pos_label=1),
        'total_samples': len(y_true)
    }

    return metrics

# Example: Calculate metrics for batch 0
metrics = calculate_batch_metrics('ModernBert_wl', 'charbot', 0)
print(f"Batch 0 Performance:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  F1-Score:  {metrics['f1']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
```

### Aggregate Results Across All Batches

```python
import gzip
import csv
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def aggregate_family_performance(model, family, num_batches=30):
    """Aggregate metrics across all batches for a family"""
    all_metrics = []

    for batch in range(num_batches):
        file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

        try:
            y_true = []
            y_pred = []

            with gzip.open(file_path, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    y_true.append(int(row['true_label']))
                    y_pred.append(int(row['predicted_label']))

            batch_metrics = {
                'f1': f1_score(y_true, y_pred, pos_label=1),
                'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            }
            all_metrics.append(batch_metrics)

        except FileNotFoundError:
            continue

    # Calculate mean and std across batches
    summary = {}
    for metric in ['f1', 'precision', 'recall']:
        values = [m[metric] for m in all_metrics]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)

    return summary

# Example: Aggregate all batches for charbot
summary = aggregate_family_performance('ModernBert_wl', 'charbot')
print(f"Charbot Overall Performance:")
print(f"  F1-Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
print(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
print(f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
```

### Compare Models on Specific Family

```python
import gzip
import csv
from sklearn.metrics import f1_score

def compare_models_on_family(models, family, batch=0):
    """Compare multiple models on the same family/batch"""
    results = {}

    for model in models:
        file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

        y_true = []
        y_pred = []

        try:
            with gzip.open(file_path, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    y_true.append(int(row['true_label']))
                    y_pred.append(int(row['predicted_label']))

            f1 = f1_score(y_true, y_pred, pos_label=1)
            results[model] = f1
        except FileNotFoundError:
            results[model] = None

    return results

# Example: Compare all models on charbot batch 0
models = [
    'ModernBert_wl',
    'ModernBert_wl_54families',
    'FANCI_wl',
    'CNN_wl',
    'gemma3_8B_wl'
]

comparison = compare_models_on_family(models, 'charbot', batch=0)
print("Model Comparison on charbot (Batch 0):")
for model, f1 in sorted(comparison.items(), key=lambda x: x[1] or 0, reverse=True):
    if f1 is not None:
        print(f"  {model:30s}: {f1:.4f}")
```

### Extract Low-Confidence Predictions

```python
import gzip
import csv

def find_low_confidence_predictions(model, family, batch, threshold=0.7):
    """Find predictions with low confidence scores"""
    file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

    low_confidence = []

    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidence = float(row['confidence_score'])
            if confidence < threshold:
                low_confidence.append({
                    'domain': row['domain'],
                    'true_label': int(row['true_label']),
                    'predicted_label': int(row['predicted_label']),
                    'confidence': confidence
                })

    return low_confidence

# Example: Find uncertain predictions
uncertain = find_low_confidence_predictions('ModernBert_wl', 'charbot', 0, threshold=0.7)
print(f"Found {len(uncertain)} predictions with confidence < 0.7")

if uncertain:
    print("\nMost uncertain predictions:")
    for pred in sorted(uncertain, key=lambda x: x['confidence'])[:10]:
        label = "DGA" if pred['true_label'] == 1 else "Benign"
        print(f"  {pred['domain']:40s} ({label}) - Confidence: {pred['confidence']:.4f}")
```

---

## 🔍 Error Analysis Use Cases

### 1. Identify Hard-to-Classify Domains

Find domains that are consistently misclassified across multiple batches:

```python
import gzip
import csv
from collections import defaultdict

def find_hard_domains(model, family, num_batches=30):
    """Find domains that appear in errors across batches"""
    domain_errors = defaultdict(int)

    for batch in range(num_batches):
        file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

        try:
            with gzip.open(file_path, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['true_label'] != row['predicted_label']:
                        domain_errors[row['domain']] += 1
        except FileNotFoundError:
            continue

    # Sort by frequency
    hard_domains = sorted(domain_errors.items(), key=lambda x: x[1], reverse=True)
    return hard_domains

# Find consistently problematic domains
hard = find_hard_domains('ModernBert_wl', 'charbot')
print("Most frequently misclassified domains:")
for domain, count in hard[:10]:
    print(f"  {domain:40s}: {count} errors across batches")
```

### 2. Confidence Distribution Analysis

Analyze confidence score distributions for correct vs. incorrect predictions:

```python
import gzip
import csv
import numpy as np

def analyze_confidence_distribution(model, family, batch):
    """Compare confidence scores for correct vs incorrect predictions"""
    correct_confidence = []
    incorrect_confidence = []

    file_path = f'Result_File/results_{model}/results_{model}_{family}.gz_{batch}.csv.gz'

    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidence = float(row['confidence_score'])
            if row['true_label'] == row['predicted_label']:
                correct_confidence.append(confidence)
            else:
                incorrect_confidence.append(confidence)

    return {
        'correct_mean': np.mean(correct_confidence),
        'correct_std': np.std(correct_confidence),
        'incorrect_mean': np.mean(incorrect_confidence),
        'incorrect_std': np.std(incorrect_confidence)
    }

# Analyze confidence
dist = analyze_confidence_distribution('ModernBert_wl', 'charbot', 0)
print("Confidence Distribution:")
print(f"  Correct predictions:   {dist['correct_mean']:.4f} ± {dist['correct_std']:.4f}")
print(f"  Incorrect predictions: {dist['incorrect_mean']:.4f} ± {dist['incorrect_std']:.4f}")
```

---

## 📈 Relationship to Aggregated Results

These detailed files are the **source data** for the aggregated metrics in [`../Result_csv/`](../Result_csv/):

**Aggregation Process:**

1. **Per-Batch Metrics:** Each compressed file generates one set of metrics
2. **Batch Aggregation:** 30 batches → mean ± std across batches
3. **Summary Files:** Aggregated results stored in `Result_csv/`

**Example:**

```
Result_File/results_ModernBert_wl_charbot.gz_0.csv.gz   →  Batch 0: F1=0.89
Result_File/results_ModernBert_wl_charbot.gz_1.csv.gz   →  Batch 1: F1=0.87
...
Result_File/results_ModernBert_wl_charbot.gz_29.csv.gz  →  Batch 29: F1=0.91

                            ↓ Aggregate

Result_csv/ModernBERT_DGA_WL_metrics_summary.csv:
  charbot,0.867,0.030,...  (mean=0.867, std=0.030)
```

---

## 📊 Data Statistics

### Per-Model Storage

| Model | Files | Compressed Size | Avg per File |
|-------|-------|-----------------|--------------|
| ModernBERT Expert | 330 | 1.8 MB | 5.5 KB |
| ModernBERT Generalist | 330 | 1.8 MB | 5.5 KB |
| DomBertUrl | 330 | 1.7 MB | 5.2 KB |
| Gemma 3 4B | 330 | 1.8 MB | 5.5 KB |
| LLaMA 3.2 3B | 330 | 1.8 MB | 5.5 KB |
| CNN | 330 | 1.7 MB | 5.2 KB |
| FANCI | 330 | 1.7 MB | 5.2 KB |
| LABin | 330 | 1.7 MB | 5.2 KB |

**Total:** 2,640 files, 14 MB compressed

---

## 🔧 Loading Utilities

### Batch Result Loader Class

```python
import gzip
import csv
from pathlib import Path

class BatchResultLoader:
    """Utility class for loading batch results"""

    def __init__(self, base_dir='Result_File'):
        self.base_dir = Path(base_dir)

    def load_batch(self, model, family, batch):
        """Load a single batch result"""
        file_path = self.base_dir / f'results_{model}' / f'results_{model}_{family}.gz_{batch}.csv.gz'

        results = []
        with gzip.open(file_path, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({
                    'domain': row['domain'],
                    'true_label': int(row['true_label']),
                    'predicted_label': int(row['predicted_label']),
                    'confidence_score': float(row['confidence_score'])
                })
        return results

    def load_all_batches(self, model, family, num_batches=30):
        """Load all batches for a family"""
        all_results = []
        for batch in range(num_batches):
            try:
                batch_results = self.load_batch(model, family, batch)
                all_results.extend(batch_results)
            except FileNotFoundError:
                continue
        return all_results

    def get_available_families(self, model):
        """Get list of available families for a model"""
        model_dir = self.base_dir / f'results_{model}'
        families = set()
        for file in model_dir.glob('*.csv.gz'):
            # Extract family from filename: results_{model}_{family}.gz_{batch}.csv.gz
            parts = file.stem.replace('.gz', '').split('_')
            # Remove model name parts and batch number
            family = '_'.join(parts[len(model.split('_')):]).rsplit('_', 1)[0]
            families.add(family)
        return sorted(families)

# Example usage
loader = BatchResultLoader()

# Load single batch
batch_0 = loader.load_batch('ModernBert_wl', 'charbot', 0)
print(f"Loaded {len(batch_0)} predictions from batch 0")

# Load all batches
all_charbot = loader.load_all_batches('ModernBert_wl', 'charbot')
print(f"Total predictions across all batches: {len(all_charbot)}")

# Get available families
families = loader.get_available_families('ModernBert_wl')
print(f"Available families: {families}")
```

---

## 📖 Related Files

- **Aggregated Metrics:** [`../Result_csv/`](../Result_csv/) - Summary statistics per family
- **Training Notebooks:** [`../Notebook/`](../Notebook/) - Model training and evaluation scripts
- **Models:** [`../Models/`](../Models/) - Model documentation and HuggingFace links
- **Datasets:** [`../Dataset/`](../Dataset/) - Training and test data

---

## 🔍 Common Tasks

### Reproduce Summary Metrics

To verify aggregated results in `Result_csv/`, aggregate these detailed files:

```python
# See "Aggregate Results Across All Batches" example above
summary = aggregate_family_performance('ModernBert_wl', 'charbot')
```

### Find Model Weaknesses

Identify families/patterns where model struggles:

```python
# Calculate F1 per family, find lowest
families = ['charbot', 'deception', 'gozi', 'manuelita', 'matsnu',
            'nymaim', 'rovnix', 'suppobox']

performance = {}
for family in families:
    summary = aggregate_family_performance('ModernBert_wl', family)
    performance[family] = summary['f1_mean']

worst_family = min(performance, key=performance.get)
print(f"Weakest performance: {worst_family} (F1={performance[worst_family]:.4f})")
```

### Statistical Significance Testing

Test if differences between models are significant:

```python
from scipy import stats

def compare_models_statistically(model1, model2, family, num_batches=30):
    """Compare two models using paired t-test"""
    model1_f1 = []
    model2_f1 = []

    for batch in range(num_batches):
        # Load and calculate F1 for both models
        # ... (use previous examples)
        pass

    t_stat, p_value = stats.ttest_rel(model1_f1, model2_f1)
    return {'t_statistic': t_stat, 'p_value': p_value}
```

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
