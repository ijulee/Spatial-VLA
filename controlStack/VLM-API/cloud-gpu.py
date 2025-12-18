from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Optional
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import functools
# Create the FastAPI app (this is your web server)
app = FastAPI(title="Robot VLM Server")
executor = ThreadPoolExecutor(max_workers=1)  # Only 1 inference at a time

# Global variables for model (loaded once at startup)
model = None
processor = None

@app.on_event("startup")
async def load_model():
    """Load model when server starts (runs once)"""
    global model, processor
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    graid_path = "./llava-graid-lora"
    # model = LlavaNextForConditionalGeneration.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda",
    #     # token=token  # Pass token explicitly
    # )
    # processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    processor = LlavaNextProcessor.from_pretrained(graid_path,use_fast=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(model, graid_path)
    model.eval()
    
prefixes_to_remove = [
            "ASSISTANT:",
            "Assistant:",
            "assistant:",
            "[INST]",
            "</s>",
            "<s>",
        ]
        
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
def run_inference_sync(request: InferenceRequest):
    """Main endpoint - robot sends images here"""
    try:
        
        # Decode base64 -> OpenCV -> RGB -> PIL Image
        img_bgr = base64_to_opencv(request.image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # print(img_rgb.shape)
        # Convert to PIL Image (what the processor expects)
        from PIL import Image
        pil_image = Image.fromarray(img_rgb)
        

        # Prepare messages for Llama 3.2 Vision
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": request.prompt}
                ]
            }
        ]
        
        # Apply chat template to get the formatted prompt
        input_text = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Process image and text separately, then combine
        inputs = processor(
            images=[pil_image],
            text=[input_text],
            return_tensors="pt"
        ).to("cuda")
        
        # Move inputs to GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False
            )
        
        # Decode output
        # generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove prompt)
        # The response usually comes after "assistant" or similar marker
        # print(response_text)
        # if "assistant" in generated_text.lower():
            # response_text = generated_text.split("assistant")[-1].strip()
        prompt_ids = processor.tokenizer(input_text, return_tensors="pt")["input_ids"][0]
        gen_ids = output[0][prompt_ids.shape[0]:]  
        # response_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # response_text = response_text.split("[/INST]")[-1]
        response_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if "[/INST]" in response_text:
            # Everything after [/INST] is the actual answer
            response_text = response_text.split("[/INST]")[-1].strip()
        # else:
        #     # Fallback: just remove the input prompt
        #     response_text = generated_text.replace(input_text, "").strip()
        

        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
                print(f"🧹 Removed prefix: '{prefix}'")

        if "</s>" in response_text:
            response_text = response_text.split("</s>")[0].strip()
            print("🧹 Removed </s> suffix")
        return  InferenceResponse(
            success=True,
            text=response_text
        )
        
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err)

        return InferenceResponse(
            success=False,
            error=str(e)
        )


def base64_to_opencv(base64_string: str) -> np.ndarray:
    """Convert base64 -> OpenCV image (no file saving!)"""
    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    return img

# async def run_inference_sync():

@app.post("/inference", response_model=InferenceResponse)
async def run_inference_await(request: InferenceRequest):
    loop = asyncio.get_event_loop()
    
    try:
        # Run blocking inference in thread pool with timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                functools.partial(run_inference_sync, request)
            ),
            timeout=4.0
        )
    except:
        print("ERROR")
    return result



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Robot VLM Server",
        "endpoints": {
            "POST /inference": "Send images for VLM processing",
        }
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    # This starts the web server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)