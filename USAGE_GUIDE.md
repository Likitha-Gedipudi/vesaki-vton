# Vesaki-VTON Usage Guide

Complete guide for using Vesaki-VTON from dataset creation to deployment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Building the Dataset](#building-the-dataset)
3. [Training Models](#training-models)
4. [Running Inference](#running-inference)
5. [Deploying to Production](#deploying-to-production)
6. [Using the Web Interface](#using-the-web-interface)
7. [API Usage Examples](#api-usage-examples)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### For First-Time Users

```bash
# 1. Install dependencies
pip3 install -r requirements_advanced.txt

# 2. Download pre-trained models (SCHP)
python3 scripts/download_models.py

# 3. Build dataset (optional if you have pre-built dataset)
python3 main_advanced.py

# 4. Train models (2-4 days on GPU)
python3 train.py --stage both --config configs/train_config.yaml

# 5. Start API server
python3 api_server.py

# 6. Start web interface
cd web && npm install && npm run dev
```

Access web interface at `http://localhost:3000`

## Building the Dataset

### Step 1: Configure URLs

Edit `product_urls.txt` with Google Shopping URLs:

```
https://www.google.com/search?q=woman+wearing+dress&tbm=shop&udm=28
https://www.google.com/search?q=model+wearing+top&tbm=shop&udm=28
```

### Step 2: Run Dataset Builder

```bash
python3 main_advanced.py
```

This will:
- Scrape 2000+ images from Google Shopping
- Classify into person/garment
- Apply data augmentation (5-8x)
- Generate all annotations (SCHP, pose, masks)
- Filter low-quality images
- Export training metadata

**Time**: 2-4 hours (GPU) or 4-8 hours (CPU)

### Step 3: Validate Dataset

```bash
python3 scripts/validate_advanced.py --data_dir dataset/train
```

Expected output:
- Person images: 500-800+ (after augmentation)
- Garment images: 500-1000+
- Annotation coverage: 85-95%

For detailed dataset documentation, see [DATASET.md](DATASET.md).

## Training Models

### Stage 1: Train GMM (Geometric Matching)

```bash
python3 train.py --stage gmm --config configs/train_config.yaml
```

**What it does**: Learns to spatially align garment with person pose

**Duration**: 6-12 hours (30 epochs on RTX 3090)

**Output**: `checkpoints/gmm_final.pth`

**Monitor progress**:
```bash
tensorboard --logdir logs/gmm
```

### Stage 2: Train TOM (Try-On Module)

```bash
python3 train.py --stage tom --config configs/train_config.yaml
```

**What it does**: Generates photorealistic try-on result

**Duration**: 14-28 hours (70 epochs on RTX 3090)

**Output**: `checkpoints/tom_final.pth`

**Monitor progress**:
```bash
tensorboard --logdir logs/tom
```

### Train Both Stages

```bash
python3 train.py --stage both --config configs/train_config.yaml
```

Total time: ~1-2 days on single GPU

### Training Configuration

Edit `configs/train_config.yaml` to adjust:

```yaml
training:
  batch_size: 4          # Reduce if out of memory
  num_epochs_gmm: 30
  num_epochs_tom: 70
  learning_rate_gmm: 0.0001
  learning_rate_tom: 0.0001

loss:
  tom_l1_weight: 1.0
  tom_perceptual_weight: 1.0
  tom_style_weight: 100.0
```

### Resume Training

```bash
python3 train.py \
  --stage tom \
  --config configs/train_config.yaml \
  --resume checkpoints/tom_epoch_35.pth
```

## Running Inference

### Single Image Inference

```bash
python3 inference.py \
  --person dataset/train/person/person001.jpg \
  --garment dataset/train/garment/garment001.jpg \
  --output result.jpg \
  --gmm_checkpoint checkpoints/gmm_final.pth \
  --tom_checkpoint checkpoints/tom_final.pth
```

### Batch Processing

Process multiple images:

```bash
# Create a batch script
for person in dataset/train/person/*.jpg; do
  for garment in dataset/train/garment/{garment1,garment2,garment3}.jpg; do
    output="results/$(basename $person .jpg)_$(basename $garment)"
    python3 inference.py --person "$person" --garment "$garment" --output "$output"
  done
done
```

### CPU vs GPU

```bash
# Force CPU inference
python3 inference.py --person person.jpg --garment garment.jpg --output result.jpg --device cpu

# Use GPU (default if available)
python3 inference.py --person person.jpg --garment garment.jpg --output result.jpg --device cuda
```

## Deploying to Production

### Local Development

```bash
# Start API server
python3 api_server.py

# In another terminal, start web interface
cd web
npm run dev
```

- API: `http://localhost:8000`
- Web: `http://localhost:3000`

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f vesaki-vton

# Stop services
docker-compose down
```

Access at `http://localhost`

### Docker with GPU

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Kubernetes Deployment

```bash
# Create cluster (if needed)
kubectl create namespace vesaki

# Deploy persistent volumes
kubectl apply -f k8s/pvc.yaml -n vesaki

# Deploy application
kubectl apply -f k8s/deployment.yaml -n vesaki

# Configure ingress
kubectl apply -f k8s/ingress.yaml -n vesaki

# Check status
kubectl get pods -n vesaki
kubectl logs -f deployment/vesaki-vton -n vesaki
```

For detailed deployment guide, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Using the Web Interface

### Development Mode

```bash
cd web
npm run dev
```

Access at `http://localhost:3000`

### Features

1. **Upload Person Image**
   - Click left upload area
   - Select image (JPG/PNG)
   - Preview appears

2. **Upload Garment Image**
   - Click right upload area
   - Select garment image
   - Preview appears

3. **Try On**
   - Click "Try On" button
   - Wait for processing (1-5 seconds)
   - View result

4. **Download Result**
   - Click "Download Result"
   - Saves as `vesaki-vton-result.jpg`

5. **Try Another**
   - Click "Try Another"
   - Upload new images

### Production Build

```bash
cd web
npm run build
```

Outputs optimized bundle to `web/dist/`

### Deploy Web Interface

**Option 1: Static Hosting** (Netlify, Vercel)
```bash
cd web
npm run build
# Upload dist/ folder to hosting provider
```

**Option 2: Docker**
```bash
cd web
docker build -t vesaki-vton-web -f Dockerfile.web .
docker run -p 3000:80 vesaki-vton-web
```

**Option 3: Nginx**
```bash
cd web
npm run build
# Copy dist/ to nginx web root
cp -r dist/* /var/www/html/
```

## API Usage Examples

### Python

```python
import requests

url = "http://localhost:8000/api/v1/tryon"

files = {
    'person_image': open('person.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open('result.jpg', 'wb') as f:
        f.write(response.content)
    print("Success!")
```

### JavaScript

```javascript
const formData = new FormData()
formData.append('person_image', personFile)
formData.append('garment_image', garmentFile)

fetch('http://localhost:8000/api/v1/tryon', {
  method: 'POST',
  body: formData
})
.then(response => response.blob())
.then(blob => {
  const url = URL.createObjectURL(blob)
  document.getElementById('result').src = url
})
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/tryon \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -o result.jpg
```

See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation.

## Model Optimization

### Export to ONNX

```bash
python3 optimize_model.py \
  --gmm_checkpoint checkpoints/gmm_final.pth \
  --tom_checkpoint checkpoints/tom_final.pth \
  --export_onnx
```

### Quantization (4x compression)

```bash
python3 optimize_model.py --quantize
```

### FP16 Conversion (2x faster on GPU)

```bash
python3 optimize_model.py --fp16
```

### Benchmark Performance

```bash
python3 optimize_model.py --benchmark
```

## Troubleshooting

### Common Issues

**1. Training stops with Out of Memory**
```bash
# Reduce batch size in configs/train_config.yaml
batch_size: 2  # or even 1
```

**2. API server won't start**
```bash
# Check if models are loaded
python3 -c "import torch; from models import GeometricMatchingModule; print('OK')"

# Check port availability
lsof -i :8000
```

**3. Web interface can't connect to API**
```bash
# Check API is running
curl http://localhost:8000/api/v1/health

# Check CORS (if needed, add to api_server.py):
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

**4. Low quality results**
- Train for more epochs
- Increase dataset size
- Adjust loss weights
- Check input image quality

**5. Slow inference**
```bash
# Use model optimization
python3 optimize_model.py --fp16 --quantize

# Use GPU instead of CPU
python3 api_server.py --device cuda
```

## Performance Tips

### Training

- Use GPU with 8GB+ VRAM
- Enable mixed precision training (AMP)
- Use batch size 4-8 for optimal GPU utilization
- Monitor with TensorBoard
- Save checkpoints frequently

### Inference

- Use FP16 on GPU (2x faster)
- Batch multiple requests
- Cache frequent person-garment pairs
- Use ONNX for deployment
- Enable GPU in Docker

### Web Interface

- Compress images before upload
- Show loading indicator
- Cache results client-side
- Use CDN for static assets
- Implement progressive image loading

## Examples

### Example 1: E-commerce Integration

```python
# Integrate into product page
import requests

def generate_tryon(customer_photo, product_id):
    garment_url = f"https://cdn.example.com/products/{product_id}.jpg"
    
    response = requests.post('http://api.example.com/api/v1/tryon', files={
        'person_image': open(customer_photo, 'rb'),
        'garment_image': requests.get(garment_url).content
    })
    
    return response.content  # Display on product page
```

### Example 2: Batch Processing for Catalog

```bash
#!/bin/bash
# Generate try-on images for entire catalog

for model in models/*.jpg; do
  for product in catalog/*.jpg; do
    python3 inference.py \
      --person "$model" \
      --garment "$product" \
      --output "catalog_results/$(basename $model .jpg)_$(basename $product)"
  done
done
```

### Example 3: Real-time Preview

```javascript
// Webcam + real-time try-on
const webcam = document.getElementById('webcam')
const canvas = document.getElementById('canvas')

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    webcam.srcObject = stream
    
    // Capture frame every 2 seconds
    setInterval(async () => {
      canvas.getContext('2d').drawImage(webcam, 0, 0)
      const blob = await new Promise(resolve => canvas.toBlob(resolve))
      
      const formData = new FormData()
      formData.append('person_image', blob)
      formData.append('garment_image', selectedGarment)
      
      const result = await fetch('/api/v1/tryon', {
        method: 'POST',
        body: formData
      })
      
      // Display result
      const resultBlob = await result.blob()
      document.getElementById('result').src = URL.createObjectURL(resultBlob)
    }, 2000)
  })
```

## Best Practices

### Dataset

- Use 500+ person images for good results
- Apply data augmentation
- Filter low-quality images
- Balance dataset (diverse poses, garments)

### Training

- Start with GMM, then TOM
- Monitor validation metrics
- Save checkpoints frequently
- Use early stopping if needed
- Fine-tune hyperparameters

### Deployment

- Use GPU for production inference
- Implement caching (Redis)
- Enable rate limiting
- Monitor performance
- Set up auto-scaling
- Use CDN for static assets

### Security

- Implement API authentication
- Rate limit endpoints
- Validate file uploads
- Sanitize inputs
- Use HTTPS in production
- Keep dependencies updated

## Support

- Documentation: [README.md](README.md)
- Dataset Guide: [DATASET.md](DATASET.md)
- API Reference: [API_REFERENCE.md](API_REFERENCE.md)
- Deployment: [DEPLOYMENT.md](DEPLOYMENT.md)
- Issues: GitHub Issues

---

**Version**: 1.0  
**Project**: Vesaki-VTON  
**Status**: Production Ready

