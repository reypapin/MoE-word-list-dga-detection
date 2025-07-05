# Models Directory

This directory contains pre-trained models for DGA detection using various machine learning approaches.

## Model Types

### BERT-based Models
- **DomBertUrl/**: Fine-tuned BERT model for domain classification
- **modernbert-dga-detector/**: Base ModernBERT model
- **modernbert-dga-detector-16families/**: ModernBERT trained on 16 DGA families
- **modernbert-dga-detector-46familias/**: ModernBERT trained on 46 DGA families

### Large Language Models
- **gemma_dga_detector/**: Gemma model fine-tuned for DGA detection
- **gemma_2epoch_dector_dga/**: Gemma model with 2-epoch training
- **llama3.2_3B_dectector_dga/**: Llama 3.2 3B model for DGA detection

### Traditional & Deep Learning Models
- **FANCI_model/**: FANCI classifier models with metadata
- **LABIN/**: LABin neural network model
- **dga_RF/**: Random Forest model with dictionaries and metadata
- **dga_cnn_model_wl/**: CNN model for word-list based DGA detection

## Model Files

Each model directory typically contains:
- Configuration files (config.json, adapter_config.json)
- Model weights (.safetensors, .pth, .keras, .joblib)
- Tokenizer files (tokenizer.json, special_tokens_map.json)
- Metadata files for model information

## Usage

Load models using the appropriate framework:
- Transformers models: Use HuggingFace Transformers library
- PyTorch models: Load .pth files with torch.load()
- Keras models: Load .keras files with tf.keras.models.load_model()
- Scikit-learn models: Load .joblib files with joblib.load()

## Performance

Refer to the evaluation results in `../Result_csv/` for model performance comparisons across different DGA families.