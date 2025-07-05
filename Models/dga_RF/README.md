# Random Forest DGA Detector - Baseline Model

This directory contains the Random Forest classifier used as a traditional machine learning baseline in our expert model evaluation study. This model demonstrates the performance of classical ensemble methods on wordlist-based DGA detection.

## Model Overview

The Random Forest model provides a robust baseline using ensemble decision trees with engineered features. It serves as a reference point for comparing the effectiveness of modern transformer-based approaches against established machine learning methods.

## Model Performance

### Key Characteristics
- **Algorithm**: Random Forest (ensemble of decision trees)
- **Features**: Word-list based engineered features
- **Performance Role**: Traditional ML baseline in comparative study
- **Inference Speed**: ~5ms per domain (very fast)
- **Throughput**: ~200,000 domains/second
- **Memory Requirements**: Minimal (~100MB)

## Files Included

### Model Files
- `dga_random_forest_model.joblib`: Trained Random Forest classifier (~50MB - excluded)
- `model_metadata.pkl`: Model training metadata and configuration
- `dga_dictionaries.pkl`: Feature extraction dictionaries and word lists

### Available Files
- `model_metadata.pkl`: Contains training parameters, feature descriptions, and performance metrics
- `dga_dictionaries.pkl`: Word lists and dictionaries used for feature engineering

## Feature Engineering

### Word-list Based Features
The Random Forest model uses carefully engineered features specifically designed for wordlist-based DGA detection:

#### Dictionary Features
- **English word presence**: Overlap with common English dictionaries
- **Technical term presence**: IT/cybersecurity related terms
- **Brand name similarity**: Similarity to legitimate brand names
- **TLD patterns**: Top-level domain usage patterns

#### Linguistic Features
- **N-gram frequencies**: Character and word n-gram patterns
- **Vowel/consonant ratios**: Linguistic balance measures
- **Syllable patterns**: Pronunciation-based features
- **Character transitions**: Smoothness of character sequences

#### Statistical Features
- **Length statistics**: Domain and component lengths
- **Entropy measures**: Character and substring entropy
- **Compression ratios**: Text compression effectiveness
- **Repetition patterns**: Character and substring repetitions

#### Structural Features
- **Subdomain patterns**: Number and structure of subdomains
- **Hyphen usage**: Placement and frequency of hyphens
- **Number integration**: Usage of digits in domain names
- **Case patterns**: Mixed case usage (if applicable)

## Usage Example

```python
import joblib
import pickle
import pandas as pd

# Load model metadata and dictionaries
with open('Models/dga_RF/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

with open('Models/dga_RF/dga_dictionaries.pkl', 'rb') as f:
    dictionaries = pickle.load(f)

# Display model information
print("Random Forest Model Metadata:")
print(f"Training date: {metadata.get('training_date')}")
print(f"Number of features: {metadata.get('n_features')}")
print(f"Number of trees: {metadata.get('n_estimators')}")
print(f"Training families: {metadata.get('dga_families')}")

# Example feature extraction using loaded dictionaries
def extract_rf_features(domain, dictionaries):
    features = {}
    
    # Dictionary-based features
    english_dict = dictionaries.get('english_words', set())
    tech_dict = dictionaries.get('tech_terms', set())
    
    domain_words = domain.replace('-', ' ').replace('.', ' ').split()
    
    features['english_word_ratio'] = sum(word in english_dict for word in domain_words) / len(domain_words) if domain_words else 0
    features['tech_term_presence'] = any(word in tech_dict for word in domain_words)
    features['domain_length'] = len(domain)
    features['subdomain_count'] = domain.count('.')
    features['hyphen_count'] = domain.count('-')
    
    # Statistical features
    features['char_entropy'] = calculate_entropy(domain)
    features['vowel_ratio'] = sum(c.lower() in 'aeiou' for c in domain) / len(domain)
    
    return features

def calculate_entropy(text):
    """Calculate character entropy of text"""
    import math
    from collections import Counter
    
    char_counts = Counter(text)
    length = len(text)
    entropy = -sum((count / length) * math.log2(count / length) 
                   for count in char_counts.values())
    return entropy

# Example usage
domain_example = "secure-banking-portal.com"
features = extract_rf_features(domain_example, dictionaries)
print(f"\nFeatures for '{domain_example}':")
for feature, value in features.items():
    print(f"  {feature}: {value}")

# Note: Actual model loading requires the .joblib file (see LARGE_FILES.md)
# model = joblib.load('Models/dga_RF/dga_random_forest_model.joblib')
# prediction = model.predict([list(features.values())])
```

## Model Training Details

### Training Configuration
- **Algorithm**: Random Forest with optimized hyperparameters
- **Trees**: Typically 100-500 decision trees
- **Max Depth**: Optimized to prevent overfitting
- **Feature Selection**: Top-k most informative features
- **Cross-validation**: 5-fold stratified cross-validation

### Training Data
- **Dataset Size**: 160,000 domains
- **DGA Families**: 8 wordlist-based families
- **Feature Count**: ~30-50 engineered features
- **Class Balance**: Balanced representation across families

## Performance Characteristics

### Strengths
- **Speed**: Extremely fast inference (~5ms per domain)
- **Interpretability**: Feature importance analysis available
- **Memory Efficiency**: Minimal memory requirements
- **Simplicity**: Easy to deploy and maintain
- **Robustness**: Stable performance across different datasets

### Limitations
- **Feature Engineering**: Requires manual feature design
- **Semantic Understanding**: Limited compared to transformer models
- **Generalization**: Lower performance on unseen DGA families
- **Maintenance**: Feature engineering needs updates for new threats

## Comparative Analysis

### vs. ModernBERT (Optimal Model)
- **Accuracy**: Lower F1-score but much faster inference
- **Speed**: 5× faster than ModernBERT
- **Resources**: Requires minimal computational resources
- **Deployment**: Easier deployment, no GPU requirements

### vs. Other Baselines
- **vs. CNN**: Similar speed, different feature approach
- **vs. FANCI**: Simpler feature engineering, ensemble approach
- **vs. LLMs**: Much faster but lower accuracy

## Deployment Scenarios

### Ideal Use Cases
- **High-throughput Processing**: Bulk domain analysis
- **Resource-constrained Environments**: Edge computing, IoT devices
- **Real-time Screening**: Initial filtering before expensive models
- **Interpretable Systems**: When feature importance is needed

### Integration Options
- **Standalone Deployment**: Independent DGA detection system
- **MoE Pre-filter**: Fast initial screening component
- **Ensemble Member**: Part of larger ensemble system
- **Baseline Reference**: Performance comparison standard

## Research Context

This Random Forest model serves multiple purposes in our research:

1. **Baseline Performance**: Reference point for transformer improvements
2. **Speed Comparison**: Demonstrates accuracy vs. speed trade-offs
3. **Feature Analysis**: Insights into important domain characteristics
4. **Deployment Alternative**: Practical option for resource-limited scenarios

## Citation

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={CACIC 2025},
  year={2024}
}
```

## File Access

**Note**: The main model file (`dga_random_forest_model.joblib`) is excluded due to size limitations. See `LARGE_FILES.md` in the repository root for instructions on accessing the complete model files.