import torch
from PIL import Image as img
from transformers import MllamaForConditionalGeneration, AutoProcessor,MllamaProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
    )
model.config.use_cache = False
processor = MllamaProcessor.from_pretrained(model_id)
raw_image = img.open("..\Photos\Lab_photo.jpg")
prompt = "<|image|><|begin_of_text|> Count the number of objects by type" 

inputs = processor(raw_image,prompt,return_tensors="pt").to(model.device)
output = model.generate(**inputs,max_new_tokens=512)

print("Robot:",processor.decode(output[0],skip_special_tokens=True))