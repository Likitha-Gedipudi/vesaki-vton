# Vesaki-VTON Quick Start

Get started with Vesaki-VTON in minutes.

## For Model Training

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (for GPU training)
- 8GB+ GPU VRAM recommended
- 16GB RAM minimum

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vesaki-vton.git
cd vesaki-vton

# Install dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements_advanced.txt

# Download pre-trained models
python3 scripts/download_models.py
```

### Prepare Dataset

Option 1: Use existing dataset
```bash
# Dataset already built in dataset/train/
python3 scripts/validate_advanced.py --data_dir dataset/train
```

Option 2: Build dataset from scratch
```bash
python3 main_advanced.py
```
See [DATASET.md](DATASET.md) for details.

### Start Training

```bash
# Train with default configuration
python3 train.py --config configs/train_config.yaml

# Train with custom settings
python3 train.py \
  --config configs/train_config.yaml \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --epochs 100
```

Training time: 2-4 days on single GPU (RTX 3090)

## For Inference

### Download Pre-trained Weights

```bash
# Download Vesaki-VTON trained model
# [Link to be added after training completes]
```

### Run Inference

Single image:
```bash
python3 inference.py \
  --person examples/person.jpg \
  --garment examples/garment.jpg \
  --output result.jpg \
  --checkpoint checkpoints/best_model.pth
```

Batch processing:
```bash
python3 inference.py \
  --person_dir data/persons/ \
  --garment_dir data/garments/ \
  --output_dir results/ \
  --checkpoint checkpoints/best_model.pth \
  --batch_size 8
```

### Start API Server

```bash
python3 api_server.py \
  --port 8000 \
  --checkpoint checkpoints/best_model.pth \
  --gpu 0
```

Test the API:
```bash
curl -X POST http://localhost:8000/api/v1/tryon \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -o result.jpg
```

## For Dataset Creation (Optional)

If you need to build a custom dataset:

### Step 1: Configure URLs

Edit `product_urls.txt` with Google Shopping URLs:
```
https://www.google.com/search?q=woman+wearing+dress&tbm=shop&udm=28
https://www.google.com/search?q=model+wearing+top&tbm=shop&udm=28
```

### Step 2: Run Pipeline

```bash
python3 main_advanced.py
```

Expected time: 2-4 hours (GPU) or 4-8 hours (CPU)

### Step 3: Validate

```bash
python3 scripts/validate_advanced.py --data_dir dataset/train
```

For detailed dataset documentation, see [DATASET.md](DATASET.md).

## Directory Overview

```
vesaki-vton/
├── train.py              # Model training script
├── inference.py          # Inference script
├── api_server.py         # API server
├── main_advanced.py      # Dataset builder
├── configs/              # Training configurations
├── models/               # Model architectures
├── scripts/              # Dataset pipeline scripts
├── dataset/              # Training dataset
└── checkpoints/          # Model checkpoints
```

## Next Steps

- **Training**: See main [README.md](README.md) for training details
- **API**: See [API_REFERENCE.md](API_REFERENCE.md) for API documentation
- **Dataset**: See [DATASET.md](DATASET.md) for dataset creation

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python3 train.py --config configs/train_config.yaml --batch_size 2
```

### Model Not Found

Ensure checkpoint path is correct:
```bash
ls checkpoints/
python3 inference.py --checkpoint checkpoints/[your_checkpoint].pth
```

### Dataset Validation Fails

Rebuild dataset:
```bash
python3 scripts/cleanup_dataset.py
python3 main_advanced.py
```

## Support

- Documentation: [README.md](README.md)
- Dataset Guide: [DATASET.md](DATASET.md)
- API Reference: [API_REFERENCE.md](API_REFERENCE.md)
- Issues: GitHub Issues

---

**Version**: 1.0  
**Project**: Vesaki-VTON
