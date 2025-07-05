# Large Model Files

Due to GitHub's 100MB file size limit, the following large model files are not included in this repository:

## Missing Model Files

### ModernBERT Models
- `Models/modernbert-dga-detector/model.safetensors` (>100MB)
- `Models/modernbert-dga-detector-16families/model.safetensors` (>100MB)  
- `Models/modernbert-dga-detector-46familias/model.safetensors` (>100MB)

### FANCI Models
- `Models/FANCI_model/fanci_dga_detector_20250618_164818.joblib` (>100MB)
- `Models/FANCI_model/mi_fanci_model_20250618_164352.joblib` (>100MB)

### Other Models
- `Models/LABIN/LABin_best_model_2025-05-30_15_26_47.keras` (>100MB)
- `Models/dga_cnn_model_wl/dga_cnn_model_wl.pth` (>100MB)

## How to Get the Models

### Option 1: Re-train the Models
Use the provided Jupyter notebooks in the `Notebook/` directory to retrain the models:
- `ModernBERT_base_DGA_Word.ipynb` for ModernBERT models
- `FANCI.ipynb` for FANCI models
- `Labin_wl.ipynb` for LABin model
- `CNN_Patron_WL.ipynb` for CNN model

### Option 2: Contact the Authors
For access to the pre-trained models, please open an issue in this repository.

### Option 3: Use Git LFS (For Repository Maintainers)
If you have Git LFS installed, you can track large files:
```bash
git lfs track "*.safetensors"
git lfs track "*.joblib" 
git lfs track "*.keras"
git lfs track "*.pth"
git add .gitattributes
git add Models/
git commit -m "Add large model files with Git LFS"
git push origin main
```

## Model Information

All model configurations, tokenizers, and metadata files are included in the repository. Only the large weight files are excluded.

### Available Files
- Configuration files (config.json)
- Tokenizer files (tokenizer.json, special_tokens_map.json)
- Adapter configurations (adapter_config.json)  
- Training arguments (training_args.bin)
- Metadata files (*.json)

These files contain all the information needed to recreate the model architecture and load pre-trained weights when available.