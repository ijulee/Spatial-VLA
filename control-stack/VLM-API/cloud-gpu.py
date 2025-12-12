# gpu_server.py - Run this on RunPod
# This IS your web server - FastAPI handles all HTTP/networking

import os

# Set Hugging Face cache to /workspace BEFORE importing transformers

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Optional
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Create the FastAPI app (this is your web server)
app = FastAPI(title="Robot VLM Server")

# Global variables for model (loaded once at startup)
model = None
processor = None

@app.on_event("startup")
async def load_model():
    """Load model when server starts (runs once)"""
    global model, processor

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    


# Request/Response formats
class InferenceRequest(BaseModel):
    image: str  # base64 encoded
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None

def base64_to_opencv(base64_string: str) -> np.ndarray:
    """Convert base64 -> OpenCV image (no file saving!)"""
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    return img

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Main endpoint - robot sends images here"""
    try:
        print(f"📸 Received inference request: {request.prompt[:50]}...")
        
        # Decode base64 -> OpenCV -> RGB -> PIL Image
        img_bgr = base64_to_opencv(request.image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image (what the processor expects)
        from PIL import Image
        pil_image = Image.fromarray(img_rgb)
        
        # Prepare for Llama 3.2 Vision
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": request.prompt}
                ]
            }
        ]
        
        input_text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        
        inputs = processor(
            pil_image,
            input_text,
            return_tensors="pt",
            padding=True,
            skip_special_tokens=True
        ).to(model.device)
        
        # Run inference
        print("🧠 Running VLM inference...")
        # with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True
        )
        
        # Decode output
        generated_text = processor.decode(output[0])
        response_text = generated_text.split("assistant")[-1].strip()
        
        print(f"✅ Response: {response_text[:100]}...")
        
        return InferenceResponse(
            success=True,
            text=response_text
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return InferenceResponse(
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Check if server is alive"""
    return {
        "status": "healthy",
        "model": "Llama-3.2-11B-Vision",
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Robot VLM Server",
        "endpoints": {
            "POST /inference": "Send images for VLM processing",
            "GET /health": "Health check"
        }
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    # This starts the web server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)