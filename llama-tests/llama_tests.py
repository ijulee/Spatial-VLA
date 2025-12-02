import torch
from PIL import Image as img
from transformers import MllamaForConditionalGeneration, AutoProcessor,MllamaProcessor

# model_id = "meta-llama/Llama-3.2-11B-Vision"
model_id = "neuralmagic/Llama-3.2-11B-Vision-Instruct-quantized.w4a16"

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


#STOLEN CODE: https://huggingface.co/RedHatAI/Llama-3.2-11B-Vision-Instruct-quantized.w4a16
# from transformers import AutoProcessor
# from vllm.assets.image import ImageAsset
# from vllm import LLM, SamplingParams

# # prepare model
# model_id = "neuralmagic/Llama-3.2-11B-Vision-Instruct-quantized.w4a16"
# llm = LLM(
#     model=model_id,
#     max_model_len=4096,
#     max_num_seqs=16,
#     limit_mm_per_prompt={"image": 1},
# )
# processor = AutoProcessor.from_pretrained(model_id)


# # prepare inputs
# question = "What is the content of this image?"
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": f"{question}"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(
#     messages, add_generation_prompt=True,tokenize=False
# )
# image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
# inputs = {
#     "prompt": prompt,
#     "multi_modal_data": {
#         "image": image
#     },
# }

# # generate response
# print("========== SAMPLE GENERATION ==============")
# outputs = llm.generate(inputs, SamplingParams(temperature=0.2, max_tokens=64))
# print(f"PROMPT  : {outputs[0].prompt}")
# print(f"RESPONSE: {outputs[0].outputs[0].text}")
# print("==========================================")
