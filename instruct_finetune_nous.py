# Author: Yuxuan Zhang
# Created date: 11202023

# === This script is called "from nous" because I will primarily use models from NousResearch.
# Based institute, very nice.

import wandb

# === Weights and Bias Setup
wandb.login()

print(
    "\n\n========== This is the script to instruction-train small models to answer medical diagnostic queries\n"
)
run_name = input("Type in the name of this training run to log on wandb: ")
print()
user_input_tags = input(
    "Any more tags you want to register to w&b besides 'Nous' and 'Instruction Tuning'? (Separate with ','): "
)
print()
user_input_tags = user_input_tags.split(",")
wandb.init(
    entity="medilora",
    project="medilora",
    notes=run_name,
    notes="",
    tags=["Nous", "Instruction Tuning"].extend(user_input_tags),
)

steps = input("Determine the max_steps for this run (i.e. 100 or 200): ")
print()

# ===
# most of this script follows this notebook:
# https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
# from this HF blogpost: https://huggingface.co/blog/4bit-transformers-bitsandbytes
# ===
# starting finetuning on medical datasets
# at this point, how we judge differential diagnostic is yet to be determined.

import torch
import transformers
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_id = "teknium/OpenHermes-2.5-Mistral-7B"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
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
        f"\n\ntrainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n\n"
    )


# original QLoRA paper had the setting of r=64 and alpha=16
config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}" # Change the format here as needed
    return text

data = load_dataset("yxzwayne/USMedicalLicenseExamsTextbooks")



# Depending on dataset, the processing of columns here will be different.
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=36,
        max_steps=int(steps),
        learning_rate=2e-5,
        fp16=True,
        logging_steps=2,
        optim="paged_adamw_8bit",
        report_to="wandb",
        output_dir="from_nous",
        push_to_hub=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# In a notebook, uncomment the line below.
# wandb.finish()
