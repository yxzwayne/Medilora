# Author: Yuxuan Zhang
# Created date: 11202023

import wandb

wandb.login()

run_name = input("Type in the name of this training run to log on wandb: ")
wandb.init(
    entity="medilora",
    project="medilora",
    notes=run_name
)

steps = input("Determine the max_steps for this run (i.e. 100 or 200): ")

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
# Choices:
# teknium/OpenHermes-2.5-Mistral-7B
# NousResearch/Nous-Hermes-llama-2-7b
# NousResearch/Nous-Hermes-Llama2-13b


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


# r=64 and alpha=16 was from the original QLoRA paper.
config = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


data = load_dataset("RafaelMPereira/HealthCareMagic-100k-Chat-Format-en")

# Depending on dataset, the processing of columns here will be different.
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],

    args=transformers.TrainingArguments(
        per_device_train_batch_size=128,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=int(steps),
        learning_rate=2e-4,
        fp16=True,
        logging_steps=2,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="wandb"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# In a notebook, uncomment the line below.
# wandb.finish()