# Evaluation Results Directory

This directory contains detailed evaluation results for different DGA detection models tested across multiple DGA families.

## Directory Structure

### Llama 3B 8-bit Results (`result_wl_Llama3B_8bits/`)
Comprehensive evaluation results for the Llama 3.2 3B model with 8-bit quantization across multiple DGA families:

#### Tested DGA Families
- **bigviktor**: Results files `results_Llama3_FineTuning_docker_bigviktor.gz_*.csv.gz`
- **charbot**: Results files `results_Llama3_FineTuning_docker_charbot.gz_*.csv.gz`
- **deception**: Results files `results_Llama3_FineTuning_docker_deception.gz_*.csv.gz`
- **gozi**: Results files `results_Llama3_FineTuning_docker_gozi.gz_*.csv.gz`
- **manuelita**: Results files `results_Llama3_FineTuning_docker_manuelita.gz_*.csv.gz`
- **matsnu**: Results files `results_Llama3_FineTuning_docker_matsnu.gz_*.csv.gz`
- **ngioweb**: Results files `results_Llama3_FineTuning_docker_ngioweb.gz_*.csv.gz`
- **nymaim**: Results files `results_Llama3_FineTuning_docker_nymaim.gz_*.csv.gz`
- **pizd**: Results files `results_Llama3_FineTuning_docker_pizd.gz_*.csv.gz`
- **rovnix**: Results files `results_Llama3_FineTuning_docker_rovnix.gz_*.csv.gz`
- **suppobox**: Results files `results_Llama3_FineTuning_docker_suppobox.gz_*.csv.gz`

### FANCI Word-List Results (`results_FANCI_wl/`)
Evaluation results for the FANCI classifier on word-list features:

#### Tested DGA Families
- **bigviktor**: Results files `results_FANCI_bigviktor.gz_*.csv.gz`
- **charbot**: Results files `results_FANCI_charbot.gz_*.csv.gz`
- **deception**: Results files `results_FANCI_deception.gz_*.csv.gz`

## File Format

Each result file contains:
- Detailed per-sample predictions
- Confidence scores
- True vs predicted labels
- Performance metrics per batch
- Evaluation timestamps

## File Naming Convention

```
results_[MODEL]_[DGA_FAMILY].gz_[BATCH_NUMBER].csv.gz
```

- **MODEL**: Model identifier (e.g., Llama3_FineTuning_docker, FANCI)
- **DGA_FAMILY**: Target DGA family name
- **BATCH_NUMBER**: Sequential batch number (0-29)

## Usage

### Extracting Results
```bash
# Extract a specific result file
gunzip results_Llama3_FineTuning_docker_bigviktor.gz_0.csv.gz

# View contents
head results_Llama3_FineTuning_docker_bigviktor.gz_0.csv
```

### Aggregating Results
```python
import pandas as pd
import glob

# Load all results for a specific DGA family
family = "bigviktor"
model = "Llama3_FineTuning_docker"
pattern = f"results_{model}_{family}.gz_*.csv.gz"
files = glob.glob(pattern)

# Combine all batch results
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
```

## Evaluation Metrics

Results typically include:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **ROC-AUC**: Area under the ROC curve

## DGA Family Characteristics

### High-Volume Families
- **bigviktor**, **gozi**, **nymaim**: Large-scale botnets with extensive domain generation
- **matsnu**, **rovnix**: Banking trojans with sophisticated DGA patterns

### Specialized Families
- **charbot**: Chat-based malware communication
- **deception**: Anti-analysis techniques
- **manuelita**: Regional-specific malware
- **ngioweb**: Web-based attack vectors
- **suppobox**: Support infrastructure malware

## Summary Statistics

For aggregated metrics across all families, refer to the summary files in `../Result_csv/`:
- `Llama3_8bits_metrics_summary.csv`
- Model-specific summary files