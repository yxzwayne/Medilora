import wandb
import torch
import transformers
from trl import SFTTrainer, trainer
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)

print(
    "\n\n========== This is the script to continue training small models on medical text corpus\n"
)

# === Training parameters

output_dir = "./nous_continued"
steps = 4000
run_name = "hermes-mistral-continued-2epoch"
model_id = "teknium/OpenHermes-2.5-Mistral-7B"
dataset_id = "yxzwayne/USMedicalLicenseExamsTextbooks"
hf_upload_model_id = "OpenHermes-2.5-Mistral-7B-USMedLicenseSFT0"

# === If training on large GPU that can fit the entire model in its vRAM:
from accelerate import Accelerator

device_index = Accelerator().process_index
device_map = {"": device_index}

# Then, add
# """
# model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    device_map=device_map
#    ...
# )
# """
# to the model import.


# === Main program


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


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


callbacks = [PeftSavingCallback()]


# === Load Dataset

# dataset = load_dataset(dataset_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
data = load_dataset(dataset_id)
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# TODO: Deleted data_text_field, may need to add it back.
dataset = trainer.ConstantLengthDataset(tokenizer=tokenizer, dataset=data)

# === From TRL SFT Documentation

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     load_in_8bit=True,
#     device_map="auto",
# )

# Per the documentation best practices,
# We create the PEFT model here, following the QLoRA notebook from HF


# This is for loading the base transformer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# From the LoRA paper, we should target the q and v module.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# deleted "device_map auto" because it will give error in distributed training
# Update: added back as said in:
# https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map=device_map
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)


training_args = transformers.TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    # max_steps=steps,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,
    save_steps=20,
    output_dir=output_dir,
    optim="paged_adamw_8bit",
    run_name=run_name,
    report_to="wandb",
    push_to_hub=True,
)

# === Weights and Bias Setup
wandb.login()

wandb.init(
    entity="medilora",
    project="medilora",
    name=run_name,
    tags=["Nous", "Continued Training"],
)


# Notes from SFT Documentation

# 1. For training adapters in 8bit, might need to tweak the arguments of the
# `prepare_model_for_kbit_training`` method from PEFT,
#  hence we advise users to either
#  - use prepare_in_int8_kwargs field,
#  - or create the PeftModel outside the SFTTrainer and pass it.
# The original notebook follows the second option, so I will be tweaking the script based on it.

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    # peft_config=peft_config,
    args=training_args,
    packing=True,
    callbacks=callbacks,
    max_seq_length=4096,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# === Saving the model after training
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

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto")
model = model.merge_and_unload()

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

model.push_to_hub(hf_upload_model_id, private=True)
