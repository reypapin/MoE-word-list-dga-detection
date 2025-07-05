# LABin Model - Neural Network Approach

This directory contains the LABin (LABoratory of INformatics) neural network model, which represents a deep learning approach to wordlist-based DGA detection using custom neural architectures.

## Model Overview

LABin implements a specialized neural network architecture designed for domain name analysis. This model explores the effectiveness of custom deep learning approaches compared to transformer-based methods and traditional machine learning baselines.

## Model Performance

### Architecture Details
- **Type**: Custom neural network for domain classification
- **Framework**: TensorFlow/Keras implementation
- **Training**: Specialized on wordlist-based DGA patterns
- **Focus**: Learning hierarchical domain name representations

## Files Included

### Model Files (Missing - See LARGE_FILES.md)
- `LABin_best_model_2025-05-30_15_26_47.keras`: Trained LABin model (~200MB)

### Training Information
- **Training Date**: May 30, 2025, 15:26:47
- **Format**: Keras model format (.keras)
- **Architecture**: Custom neural network layers

## LABin Architecture

### Neural Network Design
The LABin model employs a custom architecture optimized for domain name pattern recognition:

#### Input Processing
- **Domain Tokenization**: Character-level or subword tokenization
- **Sequence Encoding**: Fixed-length sequence representation
- **Embedding Layer**: Learned character/token embeddings
- **Positional Encoding**: Position-aware representations

#### Core Architecture
- **Hidden Layers**: Multiple fully connected or recurrent layers
- **Activation Functions**: ReLU, sigmoid, or custom activations
- **Regularization**: Dropout, batch normalization
- **Feature Extraction**: Hierarchical pattern learning

#### Output Layer
- **Classification Head**: Binary or multi-class classification
- **Output Activation**: Softmax for probability distribution
- **Loss Function**: Cross-entropy or custom loss

### Training Strategy
- **Optimization**: Adam or custom optimizer
- **Learning Rate**: Adaptive learning rate scheduling
- **Batch Size**: Optimized for available hardware
- **Regularization**: Early stopping, dropout

## Usage Example

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Note: Model file requires access to large files (see LARGE_FILES.md)
# model = keras.models.load_model('Models/LABIN/LABin_best_model_2025-05-30_15_26_47.keras')

# Example model architecture (representative)
def create_labin_model(vocab_size=1000, max_length=100, embedding_dim=128):
    """Create a LABin-style model architecture"""
    
    model = keras.Sequential([
        # Input and embedding layers
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # Feature extraction layers
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        
        # Classification layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# Example preprocessing
def preprocess_domain(domain, max_length=100):
    """Preprocess domain for LABin model"""
    # Character-level tokenization
    chars = list(domain.lower())
    
    # Create character-to-index mapping
    char_to_idx = {chr(i): i-96 for i in range(97, 123)}  # a-z
    char_to_idx.update({'.': 27, '-': 28, '_': 29})
    
    # Convert to indices
    indices = [char_to_idx.get(c, 0) for c in chars]
    
    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices.extend([0] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    return np.array(indices)

# Example usage
domain_example = "secure-banking-portal.com"
processed_domain = preprocess_domain(domain_example)
print(f"Processed domain shape: {processed_domain.shape}")

# Model prediction example (requires loaded model)
# prediction = model.predict(processed_domain.reshape(1, -1))
# is_dga = prediction[0][1] > 0.5
# confidence = prediction[0][1]
```

## Training Details

### Dataset Configuration
- **Training Size**: 160,000 domains from wordlist-based DGA families
- **Validation Split**: 20% held-out for validation
- **Test Set**: Separate unseen families for generalization testing
- **Class Balance**: Balanced representation across DGA families

### Training Process
1. **Data Preprocessing**: Domain tokenization and sequence encoding
2. **Model Architecture**: Custom neural network design
3. **Training Loop**: Iterative optimization with validation monitoring
4. **Model Selection**: Best checkpoint based on validation performance
5. **Evaluation**: Testing on unseen DGA families

### Training Configuration
- **Epochs**: Variable with early stopping
- **Batch Size**: Optimized for memory and convergence
- **Learning Rate**: Adaptive scheduling
- **Regularization**: Dropout and early stopping

## Performance Characteristics

### Strengths
- **Pattern Learning**: Automatic feature extraction from raw domains
- **Flexibility**: Customizable architecture for specific requirements
- **Neural Representations**: Rich learned domain representations
- **End-to-end Training**: Direct optimization for DGA detection

### Limitations
- **Computational Requirements**: More expensive than traditional ML
- **Training Complexity**: Requires neural network expertise
- **Hyperparameter Sensitivity**: Performance depends on architecture choices
- **Generalization**: May overfit to training families

## Research Context

### Role in Expert Evaluation
LABin serves as a **custom neural network baseline** in our comparative study:

1. **Custom Architecture**: Alternative to transformer-based approaches
2. **Domain-specific Design**: Neural network tailored for domain analysis
3. **Performance Comparison**: Effectiveness vs. transformers and traditional ML
4. **Computational Analysis**: Resource requirements compared to other approaches

### Comparison Insights
- **vs. ModernBERT**: Custom architecture vs. pre-trained transformers
- **vs. Traditional ML**: Neural learning vs. engineered features
- **vs. LLMs**: Specialized design vs. general-purpose models

## Integration Scenarios

### Deployment Options
- **Standalone System**: Independent DGA detection service
- **MoE Component**: Specialized expert in mixture of experts
- **Ensemble Member**: Part of larger ensemble system
- **Research Baseline**: Custom neural network reference

### Computational Requirements
- **Training**: GPU recommended for efficient training
- **Inference**: CPU/GPU flexible depending on throughput needs
- **Memory**: Moderate memory requirements
- **Latency**: Faster than transformers, slower than traditional ML

## Model Versioning

### File Naming Convention
- **Format**: `LABin_best_model_YYYY-MM-DD_HH_MM_SS.keras`
- **Current Version**: `LABin_best_model_2025-05-30_15_26_47.keras`
- **Timestamp**: Training completion time
- **Selection**: Best performing checkpoint during training

## Citation

This LABin model is part of our expert selection research:

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={CACIC 2025},
  year={2024}
}
```

## Development Notes

### Future Improvements
- **Architecture Optimization**: Explore different neural architectures
- **Attention Mechanisms**: Incorporate attention for better interpretability
- **Multi-task Learning**: Joint training on multiple DGA-related tasks
- **Transfer Learning**: Pre-training on larger domain corpora

### Known Issues
- **Model Size**: Large file size limits distribution
- **Training Time**: Longer training compared to traditional ML
- **Hyperparameter Tuning**: Requires extensive experimentation
- **Interpretability**: Limited compared to feature-based approaches

**Note**: The complete model file is excluded due to GitHub size limitations. See `LARGE_FILES.md` for access instructions.