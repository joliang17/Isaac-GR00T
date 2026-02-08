import os
CACHE_DIR = "/fs/nexus-projects/wilddiffusion/cache"
CACHE_DIR = "/fs/nexus-scratch/yliang17/Research/cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.data.embodiment_tags import EmbodimentTag
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM


DEFAULT_EAGLE_PATH = "/fs/nexus-scratch/yliang17/Research/VLA/GR00T/gr00t/model/backbone/eagle2_hg_model"
our_config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
eagle_processor = AutoProcessor.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True, use_fast=True)
eagle_tokenizer = eagle_processor.tokenizer

model_path = "nvidia/GR00T-N1.5-3B"
model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=torch.bfloat16)
eagle_model = model.backbone.eagle_model.cuda()
target_dtype = eagle_model.dtype  # Likely torch.bfloat16

# 3. Prepare your text prompt and image
prompt_text = "Describe this image in detail."
raw_image = Image.open("/fs/nexus-scratch/yliang17/Research/VLA/GR00T/cases/open_gripper_194.png").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
text_prompt = eagle_processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = eagle_processor(text=text_prompt, images=[raw_image], return_tensors="pt").to(eagle_model.device)

for key, value in inputs.items():
    if torch.is_floating_point(value):
        inputs[key] = value.to(device=eagle_model.device, dtype=target_dtype)
    else:
        inputs[key] = value.to(device=eagle_model.device)

# import pdb;pdb.set_trace()
with torch.no_grad(): generated_ids = eagle_model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.7)
print(eagle_processor.decode(generated_ids[0], skip_special_tokens=True))
# import pdb;pdb.set_trace()


model_name = "Qwen/Qwen3-1.7B"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
eagle_config = base_model.config

prompt_text = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt_text}]
text_prompt = eagle_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = eagle_processor(text=text_prompt, images=[raw_image], return_tensors="pt").to(eagle_model.device)
model_inputs = eagle_tokenizer([text_prompt], return_tensors="pt").to(model.device)

with torch.no_grad(): generated_ids = base_model.generate(**model_inputs, max_new_tokens=200,)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
print(eagle_processor.decode(output_ids, skip_special_tokens=True))

# 6. Decode the output
import pdb;pdb.set_trace()

