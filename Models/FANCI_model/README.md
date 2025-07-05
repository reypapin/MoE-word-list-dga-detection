# FANCI Model - Traditional ML Baseline

This directory contains the FANCI (Feature-based Algorithm for Network Classification and Identification) models, which serve as traditional machine learning baselines in our expert model evaluation.

## Model Overview

FANCI represents a feature-engineering approach to DGA detection, using handcrafted linguistic and statistical features rather than learned representations. This model provides insights into the effectiveness of traditional ML approaches compared to modern transformer-based methods.

## Model Performance

### Evaluation Results
- **Approach**: Traditional feature-based classification
- **Features**: Linguistic patterns, statistical measures, domain characteristics
- **Algorithm**: Ensemble of traditional ML algorithms
- **Performance**: Baseline comparison for transformer models

## Files Included

### Model Files (Missing - See LARGE_FILES.md)
- `fanci_dga_detector_20250618_164818.joblib`: Main FANCI classifier (~150MB)
- `mi_fanci_model_20250618_164352.joblib`: Alternative FANCI variant (~120MB)

### Metadata Files
- `fanci_dga_detector_20250618_164818_metadata.json`: Model configuration and training details
- `mi_fanci_model_20250618_164352_metadata.json`: Alternative model metadata

## FANCI Methodology

### Feature Engineering
The FANCI approach relies on carefully crafted features including:

#### Linguistic Features
- **Character n-grams**: Statistical patterns in character sequences
- **Word-level features**: Dictionary word presence and combinations
- **Linguistic coherence**: Measures of semantic plausibility
- **Pronunciation patterns**: Phonetic characteristic analysis

#### Statistical Features
- **Entropy measures**: Character and n-gram entropy
- **Length statistics**: Domain and subdomain length patterns
- **Character distribution**: Frequency analysis of characters
- **Structural patterns**: TLD usage and subdomain structure

#### Domain-specific Features
- **Wordlist overlap**: Overlap with common English dictionaries
- **Grammatical patterns**: Presence of grammatical structures
- **Semantic coherence**: Measures of meaningful word combinations
- **Brand similarity**: Similarity to legitimate brand names

### Classification Algorithm
- **Ensemble approach**: Combination of multiple traditional ML algorithms
- **Feature selection**: Automated selection of most discriminative features
- **Cross-validation**: Robust evaluation with multiple data splits
- **Hyperparameter tuning**: Grid search optimization

## Usage Example

```python
import joblib
import pandas as pd
import numpy as np

# Load the FANCI model (requires large files - see LARGE_FILES.md)
# model = joblib.load('Models/FANCI_model/fanci_dga_detector_20250618_164818.joblib')

# Load metadata
import json
with open('Models/FANCI_model/fanci_dga_detector_20250618_164818_metadata.json', 'r') as f:
    metadata = json.load(f)

print("Model training date:", metadata.get('training_date'))
print("Feature count:", metadata.get('feature_count'))
print("Training families:", metadata.get('dga_families'))

# Example feature extraction (simplified)
def extract_fanci_features(domain):
    features = {
        'length': len(domain),
        'char_entropy': calculate_entropy(domain),
        'vowel_ratio': sum(c in 'aeiou' for c in domain) / len(domain),
        'digit_ratio': sum(c.isdigit() for c in domain) / len(domain),
        # ... additional FANCI features
    }
    return features

# Example usage
domain = "secure-banking-portal.com"
features = extract_fanci_features(domain)
# prediction = model.predict([list(features.values())])
```

## Research Context

### Role in Expert Selection Study
FANCI serves as a **traditional ML baseline** in our comparative evaluation, representing:

1. **Feature Engineering Approach**: Manual feature crafting vs. learned representations
2. **Computational Efficiency**: Fast inference with minimal computational requirements
3. **Interpretability**: Clear understanding of decision factors
4. **Baseline Performance**: Reference point for transformer model improvements

### Performance Characteristics
- **Strengths**: Fast inference, interpretable results, low computational requirements
- **Limitations**: Manual feature engineering, limited semantic understanding
- **Use Cases**: Resource-constrained environments, interpretable predictions

## Model Training Details

### Training Dataset
- **Size**: 160,000 domains from wordlist-based DGA families
- **Families**: 8 wordlist-based DGA families plus benign domains
- **Features**: ~50+ engineered features per domain
- **Validation**: 5-fold cross-validation with family-aware splits

### Training Process
1. **Feature Engineering**: Manual extraction of linguistic and statistical features
2. **Feature Selection**: Automated selection of most discriminative features
3. **Model Training**: Ensemble training with hyperparameter optimization
4. **Validation**: Cross-validation and out-of-family testing

## Comparison with Modern Approaches

### Advantages of FANCI
- **Interpretability**: Clear understanding of decision factors
- **Speed**: Very fast inference (~1ms per domain)
- **Memory Efficiency**: Minimal memory requirements
- **Deployment Simplicity**: No GPU requirements

### Limitations vs. Transformers
- **Semantic Understanding**: Limited compared to ModernBERT
- **Generalization**: Lower performance on unseen families
- **Feature Engineering**: Requires domain expertise for feature design
- **Adaptation**: Manual effort needed for new threat patterns

## Integration Options

### Production Deployment
- **Edge Computing**: Suitable for resource-constrained environments
- **High-throughput Processing**: Excellent for batch domain analysis
- **Interpretable Systems**: When explanation of decisions is required
- **Baseline Comparison**: Reference model for evaluating improvements

### MoE Integration
- **Fast Pre-filtering**: Initial screening before expensive models
- **Interpretable Component**: Providing explainable decisions
- **Fallback Model**: When transformer models are unavailable
- **Feature Provider**: Engineered features for other models

## Citation

This FANCI implementation is part of our expert model evaluation study:

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={CACIC 2025},
  year={2024}
}
```