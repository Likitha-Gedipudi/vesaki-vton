# Vesaki-VTON Deployment Guide

Complete guide for deploying Vesaki-VTON in production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Model Optimization](#model-optimization)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Local Development

```bash
# Install dependencies
pip3 install -r requirements_advanced.txt

# Download models
python3 scripts/download_models.py

# Start API server
python3 api_server.py
```

Server will be available at `http://localhost:8000`

## Docker Deployment

### Build Docker Image

**CPU Version**:
```bash
docker build -t vesaki-vton:latest --target application .
```

**GPU Version**:
```bash
docker build -t vesaki-vton:gpu --target gpu-application .
```

### Run Container

**CPU**:
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/dataset:/app/dataset \
  --name vesaki-vton \
  vesaki-vton:latest
```

**GPU**:
```bash
docker run -d \
  -p 8000:8000 \
  --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/dataset:/app/dataset \
  --name vesaki-vton-gpu \
  vesaki-vton:gpu
```

### Docker Compose

**Start all services**:
```bash
# CPU version
docker-compose up -d

# GPU version
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

**Services included**:
- API server (port 8000)
- Nginx reverse proxy (port 80, 443)
- Redis cache (port 6379)
- Prometheus monitoring (port 9090)
- Grafana dashboards (port 3000)

**Stop services**:
```bash
docker-compose down
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- GPU nodes (for GPU deployment)
- Persistent volumes

### Deploy to Kubernetes

**1. Create persistent volumes**:
```bash
kubectl apply -f k8s/pvc.yaml
```

**2. Deploy application**:
```bash
kubectl apply -f k8s/deployment.yaml
```

**3. Configure ingress (optional)**:
```bash
# Edit k8s/ingress.yaml with your domain
kubectl apply -f k8s/ingress.yaml
```

**4. Check deployment**:
```bash
kubectl get pods
kubectl get services
kubectl logs -f <pod-name>
```

### Scaling

**Manual scaling**:
```bash
kubectl scale deployment vesaki-vton --replicas=5
```

**Auto-scaling** (already configured in deployment.yaml):
- Min replicas: 2
- Max replicas: 10
- CPU threshold: 70%
- Memory threshold: 80%

## Cloud Platforms

### AWS Deployment

**Option 1: ECS (Elastic Container Service)**

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag vesaki-vton:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/vesaki-vton:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/vesaki-vton:latest

# Create ECS task definition and service
aws ecs create-service --cluster vesaki-cluster --service-name vesaki-vton --task-definition vesaki-vton:1 --desired-count 3
```

**Option 2: EKS (Elastic Kubernetes Service)**

```bash
# Create EKS cluster
eksctl create cluster --name vesaki-cluster --region us-east-1 --nodegroup-name gpu-nodes --node-type g4dn.xlarge --nodes 2

# Deploy application
kubectl apply -f k8s/
```

### GCP Deployment

**GKE (Google Kubernetes Engine)**:

```bash
# Create GKE cluster
gcloud container clusters create vesaki-cluster \
  --zone us-central1-a \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 2

# Deploy application
kubectl apply -f k8s/
```

### Azure Deployment

**AKS (Azure Kubernetes Service)**:

```bash
# Create AKS cluster
az aks create \
  --resource-group vesaki-rg \
  --name vesaki-cluster \
  --node-vm-size Standard_NC6 \
  --node-count 2

# Deploy application
kubectl apply -f k8s/
```

## Model Optimization

### Export to ONNX

```bash
python3 optimize_model.py \
  --gmm_checkpoint checkpoints/gmm_final.pth \
  --tom_checkpoint checkpoints/tom_final.pth \
  --export_onnx \
  --output_dir checkpoints/optimized
```

### Quantization (INT8)

```bash
python3 optimize_model.py \
  --gmm_checkpoint checkpoints/gmm_final.pth \
  --tom_checkpoint checkpoints/tom_final.pth \
  --quantize \
  --output_dir checkpoints/optimized
```

**Benefits**:
- 4x smaller model size
- 2-4x faster inference (CPU)
- Minimal accuracy loss

### FP16 Conversion

```bash
python3 optimize_model.py \
  --gmm_checkpoint checkpoints/gmm_final.pth \
  --tom_checkpoint checkpoints/tom_final.pth \
  --fp16 \
  --output_dir checkpoints/optimized
```

**Benefits**:
- 2x smaller model size
- 2x faster inference (GPU with Tensor Cores)
- Minimal accuracy loss

### Benchmark Performance

```bash
python3 optimize_model.py \
  --gmm_checkpoint checkpoints/gmm_final.pth \
  --tom_checkpoint checkpoints/tom_final.pth \
  --benchmark
```

## Monitoring

### Prometheus Metrics

Access Prometheus dashboard: `http://localhost:9090`

**Key metrics**:
- Request rate
- Error rate
- Latency (p50, p95, p99)
- GPU utilization
- Memory usage

### Grafana Dashboards

Access Grafana: `http://localhost:3000`
- Default credentials: admin/admin

**Pre-configured dashboards**:
- API Performance
- GPU Monitoring
- System Resources

### Application Logs

**Docker**:
```bash
docker logs -f vesaki-vton
```

**Kubernetes**:
```bash
kubectl logs -f deployment/vesaki-vton
```

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Model info
curl http://localhost:8000/api/v1/model/info
```

## Performance Tuning

### API Server Configuration

Edit `api_server.py` or use environment variables:

```bash
export WORKERS=4                    # Number of worker processes
export MAX_REQUESTS=1000           # Requests per worker before restart
export TIMEOUT=300                 # Request timeout (seconds)
export KEEPALIVE=5                 # Keep-alive timeout
```

### Nginx Configuration

Edit `deployment/nginx.conf`:

- **Rate limiting**: Adjust `limit_req_zone` and `limit_req`
- **Upload size**: Modify `client_max_body_size`
- **Timeouts**: Tune `proxy_*_timeout` values
- **Worker processes**: Set based on CPU cores

### GPU Optimization

**CUDA settings**:
```bash
export CUDA_VISIBLE_DEVICES=0      # Specify GPU device
export CUDA_LAUNCH_BLOCKING=0      # Async kernel launches
```

**PyTorch optimization**:
```python
torch.backends.cudnn.benchmark = True  # Auto-tune kernels
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 on Ampere GPUs
```

## Security

### SSL/TLS Configuration

**Generate self-signed certificate** (development):
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/ssl/key.pem \
  -out deployment/ssl/cert.pem
```

**Production**: Use Let's Encrypt with cert-manager in Kubernetes

### API Authentication

Implement API key authentication:

```python
# Add to api_server.py
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/api/v1/tryon")
async def tryon(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401)
    # ... rest of endpoint
```

## Troubleshooting

### Common Issues

**1. Out of Memory (GPU)**:
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision (FP16)

**2. Slow Inference**:
- Enable model optimization (quantization, ONNX)
- Use GPU instead of CPU
- Batch multiple requests

**3. Container fails to start**:
```bash
# Check logs
docker logs vesaki-vton

# Common fixes:
# - Verify model checkpoints exist
# - Check GPU availability (if using GPU version)
# - Ensure sufficient memory
```

**4. High latency**:
- Check network bandwidth
- Enable caching (Redis)
- Use CDN for result delivery
- Scale horizontally

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python3 api_server.py
```

### Performance Profiling

```bash
# Profile model inference
python3 -m torch.utils.bottleneck inference.py --person person.jpg --garment garment.jpg --output result.jpg
```

## CI/CD Integration

GitHub Actions workflows are configured in `.github/workflows/`:

- `ci.yml`: Continuous integration (lint, test, build)
- `docker-publish.yml`: Automated Docker image publishing

**Required secrets**:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `GITHUB_TOKEN` (automatic)

## Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils

# Run load test
ab -n 1000 -c 10 -T 'multipart/form-data' \
   http://localhost:8000/api/v1/tryon
```

## Backup & Disaster Recovery

**Backup checkpoints**:
```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf backups/checkpoints_$TIMESTAMP.tar.gz checkpoints/
```

**Restore**:
```bash
tar -xzf backups/checkpoints_<timestamp>.tar.gz
```

## Cost Optimization

### AWS Cost Tips

- Use Spot Instances for training (70% savings)
- Use Fargate Spot for ECS (up to 70% savings)
- Enable auto-scaling to match demand
- Use S3 for model storage with lifecycle policies

### GCP Cost Tips

- Use Preemptible VMs (up to 80% savings)
- Use sustained use discounts
- Enable autoscaling
- Use Cloud Storage for artifacts

## Support

For deployment issues:
- Check logs first
- Review [README.md](README.md)
- Open GitHub issue
- Check health endpoint

---

**Version**: 1.0  
**Last Updated**: November 2024  
**Status**: Production Ready

