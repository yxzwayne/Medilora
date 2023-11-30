from transformers import AutoModelForCausalLM, AutoTokenizer, QwenTokenizer
from peft import PeftModel
import torch

# === Merging Mistral fine-tune

base_model = AutoModelForCausalLM.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B", torch_dtype=torch.bfloat16
)
tokenizer = QwenTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
peft_model_id = "Medilora/medilora-mimic-adapter"
model = PeftModel.from_pretrained(base_model, peft_model_id, torch_dtype=torch.bfloat16)
merged_model = model.merge_and_unload(progressbar=True)
merged_model.push_to_hub("Medilora/medilora-mistral-7b")
tokenizer.push_to_hub("Medilora/medilora-mistral-7b")


# === Merging Qwen-14b

base_model = AutoModelForCausalLM.from_pretrained(
    "CausalLM/14B", torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("CausalLM/14B")
peft_model_id = "Medilora/medilora_qwen_14B_completion"
model = PeftModel.from_pretrained(base_model, peft_model_id, torch_dtype=torch.bfloat16)
merged_model = model.merge_and_unload(progressbar=True)
merged_model.push_to_hub("Medilora/medilora-qwen-14b")
tokenizer.push_to_hub("Medilora/medilora-qwen-14b")
