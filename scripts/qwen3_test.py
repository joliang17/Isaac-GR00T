import os
CACHE_DIR = "/fs/nexus-projects/wilddiffusion/cache"
CACHE_DIR = "/fs/nexus-scratch/yliang17/Research/cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoProcessor
import torch

def compare_weights(m1, m2):
    m1_dict = m1.state_dict()
    m2_dict = m2.state_dict()
    
    matches, mismatches, missing = [], [], []
    
    for key in m1_dict:
        if key not in m2_dict:
            missing.append(key)
            continue
        
        if torch.equal(m1_dict[key], m2_dict[key]):
            matches.append(key)
        else:
            mismatches.append(key)
            
    return matches, mismatches, missing

# Note: This will likely show 0 matches for these specific models 
# because Qwen and Eagle use different naming conventions.

model_name = "Qwen/Qwen3-1.7B"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model1 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
eagle_config = model1.config

DEFAULT_EAGLE_PATH = "/fs/nexus-scratch/yliang17/Research/VLA/GR00T/gr00t/model/backbone/eagle2_hg_model"
config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
eagle_processor = AutoProcessor.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True, use_fast=True)
eagle_tokenizer = eagle_processor.tokenizer
our_config = config.text_config
model2 = AutoModelForCausalLM.from_config(our_config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model3 = AutoModelForCausalLM.from_config(eagle_config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
import pdb;pdb.set_trace()

compare_weights(model, model2)
compare_weights(model, model3)


# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model3.generate(**model_inputs, max_new_tokens=200)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
print("content:", content)

import pdb;pdb.set_trace()