# Author: Yuxuan Zhang
# Created date: 11202023
# Continuned training of small langauge models on medicla corpus.

# === This script is called "from nous" because I will primarily use models from NousResearch.
# Based institute, very nice.

import wandb
import torch
import transformers
from trl import SFTTrainer, is_xpu_available
from datasets import load_dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === Training Setup

output_dir = "./nous_continued"

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=36,
    max_steps=1000,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,
    save_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    run_name="openhermes-mistral-continued",
    report_to="wandb",
    push_to_hub=True
)

# From the LoRA paper, we should target the q and v module.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


model_id = (
    "teknium/OpenHermes-2.5-Mistral-7B"  # context length of hermes mistral is 4096
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map={"": Accelerator().local_process_index},
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)


data = load_dataset("yxzwayne/USMedicalLicenseExamsTextbooks")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# we use an undefined `config` here following this part:
# https://huggingface.co/docs/trl/v0.7.4/en/lora_tuning_peft#using-trl--peft-and-data-parallelism
trainer = SFTTrainer(
    config.model_name,
    train_dataset=data["train"],
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=4096,
    load_in_4bit=True,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=training_args
)

# === Weights and Bias Setup
wandb.login()

print(
    "\n\n========== This is the script to continue training small models on medical text corpus\n"
)


wandb.init(
    entity="medilora",
    project="medilora",
    name="openhermes-mistral-continued-sft",
    tags=["Nous", "Continued Training", "SFT"],
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
trainer.save_model(output_dir)

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

model.push_to_hub("OpenHermes-2.5-Mistral-7B-USMedLicenseSFT")