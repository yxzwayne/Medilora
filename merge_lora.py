from transformers import AutoModelForCausalLM, AutoTokenizer, QwenTokenizer
from peft import PeftModel
import torch

# === Merging Mistral fine-tune
base_id = "teknium/OpenHermes-2.5-Mistral-7B"
peft_id = ""
new_model_id = ""

base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_id)

model = PeftModel.from_pretrained(base_model, peft_id, torch_dtype=torch.bfloat16)
merged_model = model.merge_and_unload(progressbar=True)
merged_model.push_to_hub(new_model_id)
tokenizer.push_to_hub(new_model_id)
