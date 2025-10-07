# Datasets: Wordlist-Based DGA Detection

This directory contains all datasets used for training and evaluating expert models for wordlist-based DGA detection.

---

## 🔗 HuggingFace Repository

**All datasets are also available on HuggingFace:**
https://huggingface.co/Reynier/moe-wordlist-dga-models/tree/main/datasets

For large-scale experiments, downloading from HuggingFace is recommended.

---

## 📂 Directory Structure

```
Dataset/
├── README.md                    # This file
├── train/                       # Training datasets
│   ├── train_wl.csv            # Expert model training (160K samples)
│   └── train_1M.csv            # Generalist baseline (1.08M samples)
├── test-known/                  # Test sets for known families (724K samples)
│   ├── charbot.gz              # 11,001 samples
│   ├── deception.gz            # 30,001 samples
│   ├── gozi.gz                 # 50,212 samples
│   ├── manuelita.gz            # 20,001 samples
│   ├── matsnu.gz               # 116,480 samples
│   ├── nymaim.gz               # 217,773 samples
│   ├── rovnix.gz               # 120,351 samples
│   └── suppobox.gz             # 158,028 samples
└── test-generalization/         # Test sets for unseen families (13.6K samples)
    ├── bigviktor.gz            # 2,001 samples
    ├── ngioweb.gz              # 2,001 samples
    └── pizd.gz                 # 9,560 samples
```

**Total Size:** ~43 MB (compressed)

---

## 📊 Dataset Overview

| Dataset | Purpose | Samples | Families | Location | Size |
|---------|---------|---------|----------|----------|------|
| `train_wl.csv` | Train expert models | 160,000 | 8 wordlist DGAs + benign | `train/` | 4.5 MB |
| `train_1M.csv` | Train generalist baseline | 1,080,000 | 54 diverse DGAs + benign | `train/` | 31 MB |
| `test-known/*.gz` | Evaluate known families | 723,847 | 8 families | `test-known/` | 7.0 MB |
| `test-generalization/*.gz` | Test unseen DGAs | 13,562 | 3 families | `test-generalization/` | 84 KB |

---

## 1️⃣ Training Datasets

### 📄 train_wl.csv (Expert Model Training)

**Location:** `train/train_wl.csv`

**Purpose:** Train specialist expert models on wordlist-based DGA families

**Specifications:**
- **Total samples:** 160,000
- **Distribution:** Balanced (50% DGA, 50% benign)
- **DGA samples:** 80,000 (8 families × 10,000 each)
- **Benign samples:** 80,000 (Tranco top sites)

**Format:**
```csv
domain,family,label
google.com,legit,0
secure-login-check.com,suppobox,1
example-banking-portal.net,nymaim,1
```

**Columns:**
- `domain`: Domain name (string)
- `family`: DGA family name or "legit" for benign (string)
- `label`: Binary classification (0=benign, 1=DGA)

**Family Distribution:**

| Family | Samples | Type | Description |
|--------|---------|------|-------------|
| charbot | 10,000 | Wordlist DGA | Dictionary-based with common words |
| deception | 10,000 | Wordlist DGA | Social engineering themed |
| gozi | 10,000 | Wordlist DGA | Banking trojan C&C domains |
| manuelita | 10,000 | Wordlist DGA | Latin American campaign |
| matsnu | 10,000 | Wordlist DGA | Ransomware infrastructure |
| nymaim | 10,000 | Wordlist DGA | Hybrid wordlist/algorithmic |
| rovnix | 10,000 | Wordlist DGA | Bootkit malware |
| suppobox | 10,000 | Wordlist DGA | Credential stealer |
| legit | 80,000 | Benign | Tranco top 1M websites |

**Sources:**
- **DGA samples:** DGArchive, 360 Netlab, UMUDga
- **Benign domains:** Tranco top 1M sites (snapshot: 2024-07-05)
- **Collection period:** 2023-2024

**Usage Example:**
```python
import csv

# Load training data
with open('Dataset/train/train_wl.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = [(row['domain'], int(row['label']), row['family']) for row in reader]

print(f"Loaded {len(data):,} training samples")
# Output: Loaded 160,000 training samples
```

---

### 📄 train_1M.csv (Generalist Model Training)

**Location:** `train/train_1M.csv`

**Purpose:** Train generalist model for comparison (multi-family approach)

**Specifications:**
- **Total samples:** 1,080,000
- **DGA families:** 54 diverse families (wordlist + algorithmic + hybrid)
- **Distribution:** Multi-family balanced dataset
- **Size:** 31 MB

**DGA Family Categories:**
- **Wordlist-based:** 15 families (including the 8 from train_wl.csv)
- **Algorithmic:** 20 families (hash-based, pseudorandom)
- **Hybrid:** 19 families (combining techniques)

**Purpose:**
This dataset demonstrates the performance difference between:
- **Specialist approach:** Trained on specific DGA category (`train_wl.csv`)
- **Generalist approach:** Trained on diverse DGA families (`train_1M.csv`)

**Key Research Finding:**
> Specialist training improves F1-score by **+9.4%** on known families and **+30.2%** on unseen families compared to generalist training.

**Usage Example:**
```python
import csv

# Load generalist training data
with open('Dataset/train/train_1M.csv', 'r') as f:
    reader = csv.DictReader(f)
    families = set()
    for row in reader:
        families.add(row['family'])

print(f"Total families: {len(families)}")
# Output: Total families: 55 (54 DGA families + legit)
```

---

## 2️⃣ Test Datasets (Known Families)

**Location:** `test-known/*.gz`

**Purpose:** Evaluate model performance on training families (in-distribution test)

**Specifications:**
- **Total samples:** 723,847
- **Families:** 8 (same as training)
- **Format:** Gzipped text files (one domain per line, no labels)
- **Compression:** ~7 MB total

**Files:**

| File | Samples | Description | Size |
|------|---------|-------------|------|
| `charbot.gz` | 11,001 | Charbot DGA domains | 118 KB |
| `deception.gz` | 30,001 | Deception DGA domains | 309 KB |
| `gozi.gz` | 50,212 | Gozi banking trojan | 584 KB |
| `manuelita.gz` | 20,001 | Manuelita campaign | 165 KB |
| `matsnu.gz` | 116,480 | Matsnu ransomware | 1.2 MB |
| `nymaim.gz` | 217,773 | Nymaim malware | 2.5 MB |
| `rovnix.gz` | 120,351 | Rovnix bootkit | 1.4 MB |
| `suppobox.gz` | 158,028 | Suppobox stealer | 763 KB |

**Usage Example:**
```python
import gzip

# Load test samples for a specific family
with gzip.open('Dataset/test-known/charbot.gz', 'rt') as f:
    domains = [line.strip() for line in f]

print(f"Loaded {len(domains):,} charbot test domains")
# Output: Loaded 11,001 charbot test domains

# Example domains
print("Sample domains:")
for domain in domains[:5]:
    print(f"  - {domain}")
```

**Evaluation Protocol:**
- Random sample 100 domains per family per batch
- Repeat 30 batches for statistical significance
- Calculate metrics: Precision, Recall, F1-Score, FPR
- **Expected ModernBERT F1-score:** 86.7% ± 3.0%

**Note:** These files contain **only DGA domains** (no benign samples). Benign domains should be sampled separately from Tranco list for balanced evaluation.

---

## 3️⃣ Test Datasets (Generalization - Unseen Families)

**Location:** `test-generalization/*.gz`

**Purpose:** Test model generalization to unseen wordlist-based DGAs (out-of-distribution test)

**Specifications:**
- **Total samples:** 13,562
- **Families:** 3 (NOT in training data)
- **Format:** Gzipped text files
- **Compression:** ~84 KB total

**Files:**

| File | Samples | Description | Size |
|------|---------|-------------|------|
| `bigviktor.gz` | 2,001 | BigViktor DGA (unseen) | 18 KB |
| `ngioweb.gz` | 2,001 | NGIOWeb DGA (unseen) | 24 KB |
| `pizd.gz` | 9,560 | Pizd DGA (unseen) | 42 KB |

**Critical Insight:**
These families were **never seen during training** - they test the model's ability to:
1. Generalize to new wordlist-based DGA patterns
2. Distinguish novel DGA variants from benign domains
3. Handle zero-shot detection scenarios

**Characteristics:**
- **bigviktor:** Russian-language wordlist patterns
- **ngioweb:** Mixed alphanumeric with dictionary words
- **pizd:** Eastern European themed wordlist

**Usage Example:**
```python
import gzip

# Load generalization test set
families = ['bigviktor', 'ngioweb', 'pizd']
all_unseen = []

for family in families:
    with gzip.open(f'Dataset/test-generalization/{family}.gz', 'rt') as f:
        domains = [line.strip() for line in f]
        all_unseen.extend(domains)
        print(f"{family}: {len(domains):,} domains")

print(f"\nTotal unseen domains: {len(all_unseen):,}")
# Output:
# bigviktor: 2,001 domains
# ngioweb: 2,001 domains
# pizd: 9,560 domains
# Total unseen domains: 13,562
```

**Evaluation Protocol:**
- Test all available samples (no random sampling)
- Compare performance against known families
- **Expected ModernBERT F1-score:** 80.9% ± 4.5%
- **Performance drop:** ~6% (acceptable for zero-shot)

---

## 🔬 Evaluation Methodology

### Two-Phase Evaluation Protocol

#### Phase 1: Known Families Performance
- **Dataset:** `test-known/`
- **Goal:** Measure accuracy on training distribution
- **Families:** 8 familiar wordlist DGAs
- **Sampling:** 30 batches × 100 domains per family
- **Best Model:** ModernBERT Expert (86.7% F1)

#### Phase 2: Generalization Capability
- **Dataset:** `test-generalization/`
- **Goal:** Test robustness to novel DGA variants
- **Families:** 3 unseen wordlist DGAs
- **Sampling:** Full datasets (no random sampling)
- **Best Model:** ModernBERT Expert (80.9% F1)

### Test Batch Construction

For each evaluation batch:
1. Random sample 100 domains from each DGA family
2. Random sample 100 benign domains (Tranco)
3. Shuffle all 900 domains (8 DGA families + benign)
4. Classify and calculate metrics
5. Repeat 30 times for confidence intervals

### Metrics Calculation

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **False Positive Rate:** FP / (FP + TN)
- **Specificity:** TN / (TN + FP)

Where:
- TP = True Positives (DGA correctly classified)
- TN = True Negatives (benign correctly classified)
- FP = False Positives (benign misclassified as DGA)
- FN = False Negatives (DGA misclassified as benign)

---

## 📈 Dataset Statistics

### Training Data Balance

```
train_wl.csv (160K samples)
├── DGA (80K, 50%)
│   ├── charbot:    10K (6.25%)
│   ├── deception:  10K (6.25%)
│   ├── gozi:       10K (6.25%)
│   ├── manuelita:  10K (6.25%)
│   ├── matsnu:     10K (6.25%)
│   ├── nymaim:     10K (6.25%)
│   ├── rovnix:     10K (6.25%)
│   └── suppobox:   10K (6.25%)
└── Benign (80K, 50%)
    └── legit:      80K (50%)
```

### Test Data Distribution

```
Known Test (724K samples)
├── Small families (~15K):   charbot, manuelita
├── Medium families (~40K):  deception, gozi
└── Large families (~150K):  matsnu, nymaim, rovnix, suppobox

Generalization Test (14K samples)
├── bigviktor:   2K (14.8%)
├── ngioweb:     2K (14.8%)
└── pizd:        9.6K (70.4%)
```

### Domain Length Statistics

| Family | Avg Length | Min | Max | Median | Std Dev |
|--------|-----------|-----|-----|--------|---------|
| charbot | 18.3 | 8 | 35 | 17 | 4.2 |
| deception | 21.7 | 10 | 42 | 20 | 5.8 |
| gozi | 19.2 | 9 | 38 | 18 | 4.6 |
| manuelita | 20.1 | 11 | 40 | 19 | 5.1 |
| matsnu | 22.4 | 12 | 45 | 21 | 6.3 |
| nymaim | 17.8 | 8 | 33 | 17 | 3.9 |
| rovnix | 23.1 | 13 | 48 | 22 | 6.7 |
| suppobox | 25.6 | 15 | 52 | 24 | 7.4 |
| **legit** | **12.4** | **4** | **63** | **11** | **8.2** |
| bigviktor | 19.5 | 10 | 36 | 19 | 4.8 |
| ngioweb | 21.8 | 12 | 41 | 21 | 5.4 |
| pizd | 16.9 | 9 | 32 | 16 | 3.7 |

**Key Observation:** Wordlist-based DGAs are typically **longer** than benign domains:
- DGA average: ~20 characters
- Benign average: ~12 characters
- Difference: +65% longer

---

## 🛠️ Data Loading Utilities

### Load Training Data

```python
import csv

def load_training_data(dataset='train_wl'):
    """Load training dataset"""
    path = f'Dataset/train/{dataset}.csv'

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = {
            'domains': [],
            'labels': [],
            'families': []
        }
        for row in reader:
            data['domains'].append(row['domain'])
            data['labels'].append(int(row['label']))
            data['families'].append(row['family'])

    return data

# Usage
train_data = load_training_data('train_wl')
print(f"Loaded {len(train_data['domains']):,} samples")
```

### Load Test Data (Known Families)

```python
import gzip

def load_test_family(family):
    """Load test samples for a specific DGA family"""
    path = f'Dataset/test-known/{family}.gz'

    with gzip.open(path, 'rt') as f:
        domains = [line.strip() for line in f]

    return domains

# Usage
charbot_test = load_test_family('charbot')
print(f"Loaded {len(charbot_test):,} charbot test samples")
```

### Load Generalization Test Data

```python
import gzip

def load_generalization_data():
    """Load all unseen family test data"""
    families = ['bigviktor', 'ngioweb', 'pizd']
    data = {}

    for family in families:
        path = f'Dataset/test-generalization/{family}.gz'
        with gzip.open(path, 'rt') as f:
            data[family] = [line.strip() for line in f]

    return data

# Usage
unseen_data = load_generalization_data()
for family, domains in unseen_data.items():
    print(f"{family}: {len(domains):,} domains")
```

---

## 🔍 Data Quality

### Quality Assurance

1. **Label Accuracy**
   - DGA labels verified against threat intelligence sources
   - Benign domains filtered for high-confidence legitimacy
   - Manual verification of ambiguous cases
   - **Estimated error rate:** <0.1%

2. **Deduplication**
   - Removed duplicate domains within each family
   - Checked cross-family overlaps (minimal)
   - Ensured train/test separation

3. **Data Freshness**
   - DGA samples: 2023-2024
   - Benign domains: Tranco 2024-07-05
   - Test sets: Temporal separation from training

### Known Limitations

1. **Temporal Drift**
   - DGA patterns may evolve over time
   - Benign domains reflect 2024 web landscape
   - Recommend periodic dataset updates

2. **Distribution Shift**
   - Training and test sets temporally separated
   - Real-world deployment may encounter novel patterns
   - Generalization families deliberately different

3. **Label Ambiguity**
   - Some domains may be parked/expired
   - Benign classification based on reputation
   - DGA attribution may be approximate

---

## 🔗 Data Sources

### DGA Samples

1. **DGArchive** (Primary source)
   - URL: https://dgarchive.caad.fkie.fraunhofer.de/
   - Coverage: Historical DGA samples from malware analysis
   - Updated: Regularly maintained by Fraunhofer FKIE
   - Contribution: ~60% of DGA samples

2. **360 Netlab** (Active campaigns)
   - URL: https://data.netlab.360.com/dga/
   - Coverage: Recent DGA activity from active campaigns
   - Updated: Daily feeds
   - Contribution: ~30% of DGA samples

3. **UMUDga** (Academic dataset)
   - URL: https://data.mendeley.com/datasets/y8ph45msv8/1
   - Coverage: Curated research dataset
   - Updated: 2023
   - Contribution: ~10% of DGA samples

### Benign Domains

**Tranco List** (Research-oriented ranking)
- URL: https://tranco-list.eu/
- Version: Snapshot from 2024-07-05
- Methodology: Aggregated ranking from multiple sources (Alexa, Majestic, Umbrella)
- Coverage: Top 1M domains
- Selection: Randomly sampled from top 100K

---

## ⚠️ Usage Guidelines

### Ethical Considerations

- **Research Purpose Only:** Intended for academic research and defensive security
- **No Malicious Use:** Do not use to create or improve DGA algorithms
- **Privacy:** All domains are publicly observable (DNS traffic)
- **Attribution:** Cite original sources when publishing

### Recommended Practices

1. **Training:**
   - Use `train/train_wl.csv` for expert model training
   - Use `train/train_1M.csv` only for generalist comparison
   - Maintain 80/20 or 70/30 train/validation split if needed

2. **Evaluation:**
   - Test on both known (`test-known/`) and unseen (`test-generalization/`) families
   - Report both in-distribution and out-of-distribution metrics
   - Use 30 random batches for statistical confidence

3. **Reproducibility:**
   - Fix random seed for sampling
   - Document exact Tranco snapshot used for benign domains
   - Report all hyperparameters and preprocessing steps

---

## 📚 Related Files

- **Training Notebooks:** `../Notebook/ModernBERT_base_DGA_Word.ipynb`
- **Evaluation Scripts:** `../Notebook/Test_*.ipynb`
- **Model Weights:** Available on [HuggingFace](https://huggingface.co/Reynier/moe-wordlist-dga-models)
- **Results:** `../Result_csv/ModernBERT_DGA_WL_metrics_summary.csv`

---

## 📖 Citation

If you use these datasets in your research, please cite:

```bibtex
@article{leyva2025expert,
  title={Expert Selection for Wordlist-Based DGA Detection: A Systematic Evaluation},
  author={Leyva La O, Reynier and Catania, Carlos A. and Gonzalez, Rodrigo},
  journal={Under Review},
  year={2025}
}
```

And cite the original data sources:

```bibtex
@misc{dgarchive2024,
  title={DGArchive: A Deep Dive into Domain Generating Algorithms},
  author={Plohmann, Daniel and Yakdan, Khaled and Klatt, Michael},
  year={2024},
  publisher={Fraunhofer FKIE},
  url={https://dgarchive.caad.fkie.fraunhofer.de/}
}

@misc{netlab360,
  title={360 Netlab DGA Feed},
  author={360 Netlab},
  year={2024},
  url={https://data.netlab.360.com/dga/}
}

@misc{tranco2024,
  title={Tranco: A Research-Oriented Top Sites Ranking},
  author={Le Pochat, Victor and Van Goethem, Tom and Tajalizadehkhoob, Samaneh and Korczynski, Maciej and Joosen, Wouter},
  year={2024},
  url={https://tranco-list.eu/}
}
```

---

## 📞 Contact

For questions about the datasets:
- **Author:** Reynier Leyva La O
- **Email:** rleyvalao@mendoza-conicet.gob.ar
- **Institution:** CONICET Argentina
- **HuggingFace:** https://huggingface.co/Reynier/moe-wordlist-dga-models
- **GitHub:** https://github.com/reypapin/MoE-word-list-dga-detection

---

**Last Updated:** October 2025
