# Vesaki-VTON API Reference

Complete API documentation for the Vesaki-VTON virtual try-on service.

## Overview

The Vesaki-VTON API provides RESTful endpoints for virtual garment try-on functionality. The API accepts person and garment images, processes them through the trained model, and returns high-quality try-on results.

**Base URL**: `http://localhost:8000/api/v1`

**Protocol**: HTTP/HTTPS

**Authentication**: Optional (configure in deployment)

## Getting Started

### Starting the API Server

```bash
python3 api_server.py \
  --port 8000 \
  --checkpoint checkpoints/best_model.pth \
  --gpu 0 \
  --workers 4
```

Parameters:
- `--port`: Server port (default: 8000)
- `--checkpoint`: Path to trained model weights
- `--gpu`: GPU device ID (-1 for CPU)
- `--workers`: Number of worker processes

### Testing the Server

```bash
curl http://localhost:8000/api/v1/health
```

## Endpoints

### 1. Health Check

Check if the API server is running and model is loaded.

**Endpoint**: `GET /api/v1/health`

**Request**: No parameters required

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

**Status Codes**:
- `200 OK`: Server is healthy
- `503 Service Unavailable`: Server is starting or model not loaded

**Example**:
```bash
curl http://localhost:8000/api/v1/health
```

---

### 2. Virtual Try-On (Single Image)

Generate a virtual try-on result from a person image and a garment image.

**Endpoint**: `POST /api/v1/tryon`

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| person_image | File | Yes | Person image (JPEG/PNG) |
| garment_image | File | Yes | Garment image (JPEG/PNG) |
| output_format | String | No | Output format: "jpeg" or "png" (default: "jpeg") |
| quality | Integer | No | JPEG quality 1-100 (default: 95) |
| return_intermediate | Boolean | No | Return intermediate results (default: false) |
| generate_mask | Boolean | No | Return composition mask (default: false) |

**Response** (JSON):
```json
{
  "status": "success",
  "result_image": "base64_encoded_image_string",
  "processing_time": 1.23,
  "metadata": {
    "person_resolution": [768, 1024],
    "garment_resolution": [768, 1024],
    "output_resolution": [768, 1024],
    "model_version": "1.0.0"
  },
  "intermediate_results": {
    "warped_garment": "base64_encoded_image_string",
    "composition_mask": "base64_encoded_image_string"
  }
}
```

**Error Response**:
```json
{
  "status": "error",
  "error_code": "INVALID_IMAGE",
  "message": "Person image could not be decoded",
  "details": "Corrupted or unsupported image format"
}
```

**Status Codes**:
- `200 OK`: Success
- `400 Bad Request`: Invalid input parameters
- `413 Payload Too Large`: Image file too large (>10MB)
- `500 Internal Server Error`: Processing failed

**Example (cURL)**:
```bash
curl -X POST http://localhost:8000/api/v1/tryon \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -F "output_format=jpeg" \
  -F "quality=95" \
  -o result.jpg
```

**Example (Python)**:
```python
import requests

url = "http://localhost:8000/api/v1/tryon"

files = {
    'person_image': open('person.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}

data = {
    'output_format': 'jpeg',
    'quality': 95
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    # Decode base64 image
    import base64
    image_data = base64.b64decode(result['result_image'])
    with open('result.jpg', 'wb') as f:
        f.write(image_data)
    print(f"Processing time: {result['processing_time']}s")
else:
    print(f"Error: {response.json()['message']}")
```

**Example (JavaScript/Node.js)**:
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('person_image', fs.createReadStream('person.jpg'));
form.append('garment_image', fs.createReadStream('garment.jpg'));
form.append('output_format', 'jpeg');
form.append('quality', '95');

axios.post('http://localhost:8000/api/v1/tryon', form, {
    headers: form.getHeaders()
})
.then(response => {
    const resultImage = Buffer.from(response.data.result_image, 'base64');
    fs.writeFileSync('result.jpg', resultImage);
    console.log(`Processing time: ${response.data.processing_time}s`);
})
.catch(error => {
    console.error('Error:', error.response.data.message);
});
```

---

### 3. Batch Processing

Process multiple person-garment pairs in a single request.

**Endpoint**: `POST /api/v1/batch`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "pairs": [
    {
      "person_url": "http://example.com/person1.jpg",
      "garment_url": "http://example.com/garment1.jpg",
      "pair_id": "pair_001"
    },
    {
      "person_url": "http://example.com/person2.jpg",
      "garment_url": "http://example.com/garment2.jpg",
      "pair_id": "pair_002"
    }
  ],
  "callback_url": "http://example.com/webhook",
  "options": {
    "output_format": "jpeg",
    "quality": 95
  }
}
```

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pairs | Array | Yes | Array of person-garment pairs |
| pairs[].person_url | String | Yes | URL to person image |
| pairs[].garment_url | String | Yes | URL to garment image |
| pairs[].pair_id | String | No | Custom identifier for pair |
| callback_url | String | No | Webhook URL for completion notification |
| options | Object | No | Processing options (format, quality) |

**Response** (Immediate):
```json
{
  "status": "processing",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_pairs": 2,
  "completed": 0,
  "estimated_time_seconds": 4.5,
  "status_url": "/api/v1/batch/550e8400-e29b-41d4-a716-446655440000"
}
```

**Status Codes**:
- `202 Accepted`: Batch job created
- `400 Bad Request`: Invalid request parameters
- `429 Too Many Requests`: Rate limit exceeded

**Example**:
```python
import requests
import json

url = "http://localhost:8000/api/v1/batch"

data = {
    "pairs": [
        {
            "person_url": "http://example.com/person1.jpg",
            "garment_url": "http://example.com/garment1.jpg",
            "pair_id": "pair_001"
        },
        {
            "person_url": "http://example.com/person2.jpg",
            "garment_url": "http://example.com/garment2.jpg",
            "pair_id": "pair_002"
        }
    ],
    "callback_url": "http://example.com/webhook",
    "options": {
        "output_format": "jpeg",
        "quality": 95
    }
}

response = requests.post(url, json=data)
result = response.json()
job_id = result['job_id']
print(f"Job ID: {job_id}")
```

---

### 4. Batch Status

Check the status of a batch processing job.

**Endpoint**: `GET /api/v1/batch/{job_id}`

**Path Parameters**:
- `job_id`: UUID of the batch job

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_pairs": 2,
  "completed": 2,
  "failed": 0,
  "results": [
    {
      "pair_id": "pair_001",
      "status": "success",
      "result_url": "http://localhost:8000/results/pair_001.jpg",
      "processing_time": 1.23
    },
    {
      "pair_id": "pair_002",
      "status": "success",
      "result_url": "http://localhost:8000/results/pair_002.jpg",
      "processing_time": 1.45
    }
  ],
  "created_at": "2024-11-27T10:30:00Z",
  "completed_at": "2024-11-27T10:30:05Z"
}
```

**Job Status Values**:
- `queued`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: All pairs processed successfully
- `partial`: Some pairs failed
- `failed`: Job failed completely

**Status Codes**:
- `200 OK`: Job found
- `404 Not Found`: Job ID not found

**Example**:
```bash
curl http://localhost:8000/api/v1/batch/550e8400-e29b-41d4-a716-446655440000
```

---

### 5. Model Information

Get information about the loaded model.

**Endpoint**: `GET /api/v1/model/info`

**Response**:
```json
{
  "model_name": "Vesaki-VTON",
  "version": "1.0.0",
  "architecture": "GMM + TOM",
  "input_resolution": [768, 1024],
  "output_resolution": [768, 1024],
  "parameters": {
    "total": 45000000,
    "gmm": 15000000,
    "tom": 30000000
  },
  "training_info": {
    "dataset_size": 1500,
    "epochs": 100,
    "trained_on": "2024-11-20"
  }
}
```

**Status Codes**:
- `200 OK`: Success

**Example**:
```bash
curl http://localhost:8000/api/v1/model/info
```

---

## Error Codes

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| INVALID_IMAGE | 400 | Image file is corrupted or unsupported format |
| IMAGE_TOO_LARGE | 413 | Image file exceeds 10MB limit |
| MISSING_PARAMETER | 400 | Required parameter is missing |
| INVALID_FORMAT | 400 | Invalid value for output_format |
| PROCESSING_FAILED | 500 | Model processing failed |
| MODEL_NOT_LOADED | 503 | Model is not loaded yet |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| JOB_NOT_FOUND | 404 | Batch job ID not found |

## Rate Limiting

**Default Limits**:
- Single image endpoint: 60 requests/minute
- Batch endpoint: 10 requests/minute
- Status check endpoint: 300 requests/minute

**Rate Limit Headers**:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640000000
```

**Rate Limit Exceeded Response**:
```json
{
  "status": "error",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Try again in 30 seconds.",
  "retry_after": 30
}
```

## Image Requirements

**Supported Formats**:
- JPEG (.jpg, .jpeg)
- PNG (.png)

**Size Limits**:
- Minimum: 256x256 pixels
- Maximum: 4096x4096 pixels
- File size: 10MB maximum

**Recommendations**:
- Person image: Front-facing, full or upper body visible
- Garment image: Flat lay or mannequin shot, clear visibility
- Good lighting, minimal occlusions
- Aspect ratio close to 3:4 (width:height)

## Authentication

**API Key Authentication** (if enabled):

Include API key in request header:
```
Authorization: Bearer YOUR_API_KEY
```

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/tryon \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg"
```

## Webhooks

For batch processing, provide a `callback_url` to receive completion notifications.

**Webhook Payload** (POST request):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_pairs": 2,
  "completed": 2,
  "failed": 0,
  "results_url": "http://localhost:8000/api/v1/batch/550e8400-e29b-41d4-a716-446655440000"
}
```

**Webhook Security**:
- HMAC signature in `X-Webhook-Signature` header
- Verify signature to ensure authenticity

## SDKs and Client Libraries

### Python SDK

```python
from vesaki_vton import VesakiClient

client = VesakiClient(
    base_url="http://localhost:8000",
    api_key="YOUR_API_KEY"  # optional
)

# Single image try-on
result = client.tryon(
    person_image="person.jpg",
    garment_image="garment.jpg",
    output_path="result.jpg"
)

print(f"Processing time: {result.processing_time}s")

# Batch processing
job = client.batch_tryon([
    ("person1.jpg", "garment1.jpg"),
    ("person2.jpg", "garment2.jpg")
])

job.wait()  # Wait for completion
results = job.get_results()
```

### JavaScript SDK

```javascript
const VesakiClient = require('vesaki-vton');

const client = new VesakiClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'YOUR_API_KEY'  // optional
});

// Single image try-on
client.tryon({
    personImage: 'person.jpg',
    garmentImage: 'garment.jpg',
    outputPath: 'result.jpg'
})
.then(result => {
    console.log(`Processing time: ${result.processingTime}s`);
})
.catch(error => {
    console.error('Error:', error.message);
});
```

## Performance Tips

1. **Use batch processing** for multiple pairs to reduce overhead
2. **Resize images** before upload to reduce transfer time
3. **Cache results** on client side when possible
4. **Use JPEG format** for faster encoding (vs PNG)
5. **Enable GPU** on server for 10x faster processing
6. **Use multiple workers** for concurrent request handling

## Deployment Configurations

### Docker Deployment

```yaml
version: '3'
services:
  vesaki-vton:
    image: vesaki-vton:latest
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4
      - GPU_DEVICE=0
    volumes:
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vesaki-vton
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vesaki-vton
  template:
    metadata:
      labels:
        app: vesaki-vton
    spec:
      containers:
      - name: vesaki-vton
        image: vesaki-vton:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: WORKERS
          value: "4"
```

## Monitoring

**Health Check Endpoint**: `/api/v1/health`

**Metrics Endpoint**: `/api/v1/metrics` (Prometheus format)

**Available Metrics**:
- Request count by endpoint
- Request duration histogram
- Error rate
- GPU utilization
- Model inference time

## Support

- Documentation: [README.md](README.md)
- Issues: GitHub Issues
- Email: support@vesaki-vton.com

---

**Version**: 1.0  
**Last Updated**: November 2024  
**Project**: Vesaki-VTON

