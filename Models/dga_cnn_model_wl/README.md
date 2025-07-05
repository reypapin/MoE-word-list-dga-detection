# CNN Model for Wordlist-based DGA Detection

This directory contains a Convolutional Neural Network (CNN) model specifically designed for detecting wordlist-based Domain Generation Algorithms (DGAs). The CNN approach focuses on learning local patterns and n-gram features from domain names.

## Model Overview

The CNN model employs convolutional layers to capture local patterns in domain names, making it particularly effective at identifying character-level and n-gram patterns characteristic of different DGA families. This approach bridges traditional pattern recognition with deep learning capabilities.

## Model Performance

### Key Characteristics
- **Architecture**: Convolutional Neural Network with wordlist features
- **Pattern Focus**: Local character patterns and n-gram features
- **Inference Speed**: ~15ms per domain
- **Throughput**: ~66,000 domains/second
- **F1-Score**: Approximately 78.9% on known families, 72.1% on unknown families

## Files Included

### Model Files (Missing - See LARGE_FILES.md)
- `dga_cnn_model_wl.pth`: Trained CNN model weights (~100MB)

### Directory Structure
- `.data/serialization_id`: PyTorch model serialization metadata

## CNN Architecture

### Network Design
The CNN model is specifically architected for domain name pattern recognition:

#### Input Layer
- **Input Format**: Domain names converted to character sequences
- **Encoding**: Character-level or n-gram embeddings
- **Sequence Length**: Fixed-length padding/truncation
- **Feature Engineering**: Wordlist-based feature integration

#### Convolutional Layers
- **Conv1D Layers**: Multiple convolutional filters for pattern detection
- **Filter Sizes**: Various kernel sizes (e.g., 3, 4, 5) to capture different n-gram patterns
- **Feature Maps**: Multiple feature maps per filter size
- **Activation**: ReLU activation functions

#### Pooling and Regularization
- **Max Pooling**: Dimensionality reduction and translation invariance
- **Dropout**: Regularization to prevent overfitting
- **Batch Normalization**: Training stability and convergence

#### Classification Head
- **Dense Layers**: Fully connected layers for final classification
- **Output**: Binary classification (DGA/benign) or multi-class family prediction
- **Activation**: Softmax for probability distribution

## Usage Example

```python
import torch
import torch.nn as nn
import numpy as np

# Note: Model loading requires large files (see LARGE_FILES.md)
# model = torch.load('Models/dga_cnn_model_wl/dga_cnn_model_wl.pth')

class DGACNNModel(nn.Module):
    """CNN model for DGA detection (representative architecture)"""
    
    def __init__(self, vocab_size=100, embed_dim=128, num_classes=2):
        super(DGACNNModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        
        # Pooling and regularization
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Classification layers
        self.fc1 = nn.Linear(384, 256)  # 3 conv layers × 128 filters
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        # Convolutional feature extraction
        conv1_out = self.relu(self.conv1(embedded))
        conv2_out = self.relu(self.conv2(embedded))
        conv3_out = self.relu(self.conv3(embedded))
        
        # Global max pooling
        pool1 = self.pool(conv1_out).squeeze(2)
        pool2 = self.pool(conv2_out).squeeze(2)
        pool3 = self.pool(conv3_out).squeeze(2)
        
        # Concatenate features
        features = torch.cat([pool1, pool2, pool3], dim=1)
        features = self.dropout(features)
        
        # Classification
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return self.softmax(x)

# Domain preprocessing for CNN
def preprocess_domain_cnn(domain, max_length=100):
    """Convert domain to character indices for CNN processing"""
    
    # Character vocabulary (a-z, 0-9, ., -, _)
    char_to_idx = {}
    idx = 1  # Reserve 0 for padding
    
    # Add alphabetic characters
    for c in 'abcdefghijklmnopqrstuvwxyz':
        char_to_idx[c] = idx
        idx += 1
    
    # Add numeric characters
    for c in '0123456789':
        char_to_idx[c] = idx
        idx += 1
    
    # Add special characters
    for c in '.-_':
        char_to_idx[c] = idx
        idx += 1
    
    # Convert domain to indices
    domain_lower = domain.lower()
    indices = [char_to_idx.get(c, 0) for c in domain_lower]
    
    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices.extend([0] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    return torch.tensor(indices, dtype=torch.long)

# Example usage
domain_example = "secure-banking-portal.com"
processed_domain = preprocess_domain_cnn(domain_example)
print(f"Processed domain shape: {processed_domain.shape}")

# Model prediction (requires loaded model)
# model.eval()
# with torch.no_grad():
#     input_tensor = processed_domain.unsqueeze(0)  # Add batch dimension
#     prediction = model(input_tensor)
#     is_dga = prediction[0][1] > 0.5
#     confidence = prediction[0][1].item()
```

## Training Configuration

### Dataset Details
- **Training Data**: 160,000 domains from wordlist-based DGA families
- **Feature Integration**: Wordlist-based features combined with raw character data
- **Augmentation**: Domain variation and synthetic example generation
- **Validation**: Stratified splitting with family-aware validation

### Training Parameters
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Cross-entropy loss
- **Batch Size**: Optimized for GPU memory
- **Epochs**: Training with early stopping
- **Learning Rate**: Initial rate with decay scheduling

### Hardware Requirements
- **Training**: GPU recommended (CUDA compatible)
- **Memory**: 4-8GB GPU memory for training
- **Inference**: CPU or GPU compatible
- **Storage**: Model size ~100MB

## Performance Analysis

### Strengths
- **Pattern Recognition**: Excellent at detecting local character patterns
- **Speed**: Fast inference (~15ms per domain)
- **N-gram Features**: Captures important n-gram characteristics
- **Scalability**: Efficient for high-throughput processing

### Limitations
- **Local Patterns Only**: Limited understanding of global semantic structure
- **Sequence Dependencies**: Less effective at long-range dependencies
- **Wordlist Integration**: Challenges in incorporating semantic wordlist features
- **Generalization**: Lower performance on sophisticated wordlist DGAs

## Comparison with Other Models

### vs. ModernBERT (Optimal)
- **Speed**: 2× faster than ModernBERT
- **Accuracy**: Lower F1-score but still competitive
- **Resources**: Lower memory requirements
- **Deployment**: Easier deployment without transformer dependencies

### vs. Traditional ML (RF, FANCI)
- **Feature Learning**: Automatic pattern learning vs. manual engineering
- **Performance**: Better accuracy than traditional approaches
- **Complexity**: Higher complexity but better pattern recognition
- **Interpretability**: Less interpretable than feature-based methods

### vs. Large Language Models
- **Efficiency**: Much faster and lighter than LLMs
- **Accuracy**: Lower accuracy but practical efficiency
- **Deployment**: Easier deployment and maintenance
- **Resource Requirements**: Significantly lower resource needs

## Deployment Scenarios

### Ideal Use Cases
- **Real-time Processing**: Fast inference for live DNS monitoring
- **Edge Computing**: Deployable on edge devices with moderate compute
- **High-throughput Systems**: Bulk domain analysis
- **Balanced Performance**: Good accuracy-speed trade-off

### Integration Options
- **Standalone Service**: Independent DGA detection API
- **MoE Component**: Pattern recognition expert in MoE system
- **Preprocessing Stage**: Initial filtering before expensive models
- **Ensemble Member**: Component in multi-model ensemble

## Research Context

### Role in Expert Selection Study
The CNN model represents the **pattern recognition approach** in our comparative evaluation:

1. **Local Pattern Learning**: Effectiveness of convolutional architectures
2. **Speed-Accuracy Trade-off**: Balance between performance and efficiency
3. **Feature Learning**: Automatic vs. manual feature engineering
4. **Deployment Practicality**: Real-world deployment considerations

## Citation

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={CACIC 2025},
  year={2024}
}
```

## Model Access

**Note**: The model weights file (`dga_cnn_model_wl.pth`) is excluded due to GitHub size limitations. Refer to `LARGE_FILES.md` in the repository root for instructions on accessing the complete model files.