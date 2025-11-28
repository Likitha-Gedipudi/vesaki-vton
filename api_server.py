#!/usr/bin/env python3
"""
Vesaki-VTON API Server
FastAPI server for virtual try-on inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from inference import VITONInference

# Initialize FastAPI app
app = FastAPI(
    title="Vesaki-VTON API",
    description="Virtual Try-On System API",
    version="1.0.0"
)

# Global inference instance
inferencer = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global inferencer
    try:
        inferencer = VITONInference(
            gmm_checkpoint='checkpoints/gmm_final.pth',
            tom_checkpoint='checkpoints/tom_final.pth',
            config_path='configs/train_config.yaml',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": inferencer is not None,
        "gpu_available": torch.cuda.is_available(),
        "version": "1.0.0"
    }


@app.post("/api/v1/tryon")
async def virtual_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    """
    Virtual try-on endpoint
    
    Args:
        person_image: Person image file
        garment_image: Garment image file
    Returns:
        result_image: Try-on result
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded files
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        # Convert to PIL Images
        person_img = Image.open(io.BytesIO(person_bytes)).convert('RGB')
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert('RGB')
        
        # Save temporarily
        person_path = "temp_person.jpg"
        garment_path = "temp_garment.jpg"
        person_img.save(person_path)
        garment_img.save(garment_path)
        
        # Run inference
        result = inferencer.try_on(person_path, garment_path)
        
        # Convert result to bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "Vesaki-VTON",
        "version": "1.0.0",
        "architecture": "GMM + TOM",
        "input_resolution": [768, 1024],
        "output_resolution": [768, 1024]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

