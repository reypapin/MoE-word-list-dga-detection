# ModernBERT DGA Detector - Optimal Expert Model

This directory contains the **optimal expert model** identified in our research for wordlist-based DGA detection. ModernBERT achieved the best balance of accuracy, generalization, and computational efficiency.

## Model Performance

### Key Metrics
- **F1-Score (Known Families)**: 86.7%
- **F1-Score (Unknown Families)**: 80.9%
- **Inference Time**: 26ms on NVIDIA Tesla T4 GPU
- **Throughput**: ~38,000 domains/second
- **Model Size**: Base ModernBERT with domain-specific fine-tuning

### Performance Advantages
- **9.4% improvement** over generalist approaches on known families
- **30.2% improvement** over generalist approaches on unknown families
- **25× faster** than large language model alternatives
- Superior generalization to unseen DGA families

## Model Architecture

### Base Model
- **Architecture**: ModernBERT-base (enhanced BERT variant)
- **Parameters**: ~110M parameters
- **Input**: Domain names (tokenized)
- **Output**: Binary classification (DGA/benign) + family attribution

### Training Details
- **Training Data**: 160,000 domains from 8 wordlist-based DGA families
- **Families**: bigviktor, charbot, deception, gozi, manuelita, matsnu, ngioweb, nymaim, pizd, rovnix, suppobox
- **Training Strategy**: Domain-specific fine-tuning on wordlist patterns
- **Optimization**: AdamW optimizer with learning rate scheduling

## Files Included

### Configuration Files
- `config.json`: Model architecture configuration
- `tokenizer_config.json`: Tokenizer settings and vocabulary
- `special_tokens_map.json`: Special token mappings
- `training_args.bin`: Training hyperparameters and settings

### Tokenizer Files
- `tokenizer.json`: Fast tokenizer implementation
- `vocab.txt`: Model vocabulary (if applicable)

### Missing Files (Due to Size Limits)
- `model.safetensors`: Model weights (~450MB) - See `LARGE_FILES.md` for access

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "Models/modernbert-dga-detector/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example inference
domain = "secure-banking-portal.com"
inputs = tokenizer(domain, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    
# Get prediction
is_dga = predictions[0][1] > 0.5  # Assuming class 1 is DGA
confidence = predictions[0][1].item()

print(f"Domain: {domain}")
print(f"Is DGA: {is_dga}")
print(f"Confidence: {confidence:.3f}")
```

## Deployment Considerations

### Production Deployment
- **GPU Recommended**: NVIDIA Tesla T4 or equivalent for optimal performance
- **Memory Requirements**: ~2GB GPU memory for inference
- **Batch Processing**: Supports batch inference for higher throughput
- **Latency**: 26ms per domain (individual inference)

### Integration Options
- **REST API**: Deploy as microservice for real-time DNS monitoring
- **Batch Processing**: For large-scale domain analysis
- **Edge Deployment**: Optimized for edge computing environments
- **MoE Integration**: Designed as expert component in Mixture of Experts systems

## Research Context

This model was identified as the optimal expert through systematic evaluation of seven candidate models across multiple performance dimensions:

1. **Detection Accuracy**: Highest F1-scores on both known and unknown families
2. **Generalization**: Best performance on unseen DGA families
3. **Computational Efficiency**: Optimal inference time vs. accuracy trade-off
4. **Operational Reliability**: Low false positive rates
5. **Deployment Feasibility**: Practical memory and latency requirements

## Citation

If you use this model in your research, please cite:

```bibtex
@inproceedings{leyva2024specialized,
  title={Specialized Expert Models for Wordlist-Based DGA Detection: A Mixture of Experts Approach},
  author={Leyva La O, Reynier and Gonzalez, Rodrigo and Catania, Carlos A.},
  booktitle={CACIC 2025},
  year={2024}
}
```

## Model Limitations

### Known Challenges
- **Sophisticated Families**: Some families like `manuelita` remain challenging
- **Domain Length**: Performance may vary with very long or very short domains
- **Language Bias**: Optimized for English wordlist patterns
- **Adversarial Robustness**: May be vulnerable to specifically crafted adversarial domains

### Recommended Use Cases
- **Real-time DNS monitoring** in enterprise environments
- **Threat intelligence** for DGA family attribution
- **Research baselines** for wordlist-based DGA detection
- **Component integration** in larger cybersecurity systems