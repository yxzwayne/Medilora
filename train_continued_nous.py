import wandb
import torch
import transformers
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# === Training Setup

output_dir = "./nous_continued"

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=1000,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,
    save_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    run_name="openhermes-mistral-continued",
    report_to="wandb",
    push_to_hub=True,
)

# From the LoRA paper, we should target the q and v module.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


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


from peft import prepare_model_for_kbit_training

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

from datasets import load_dataset

data = load_dataset("yxzwayne/USMedicalLicenseExamsTextbooks")
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)


# === Weights and Bias Setup
wandb.login()

print(
    "\n\n========== This is the script to continue training small models on medical text corpus\n"
)


wandb.init(
    entity="medilora",
    project="medilora",
    name="openhermes-mistral-continued",
    tags=["Nous", "Continued Training"],
)


trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=1000,
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
trainer.save_model(output_dir)

import os

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

from trl import is_xpu_available

# Free memory for merging weights
del model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

model.push_to_hub("OpenHermes-2.5-Mistral-7B-USMedLicenseTrained")
