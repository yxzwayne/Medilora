import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator

device_index = Accelerator().process_index
device_map = {"": device_index}

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


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device_map)

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


from peft import LoraConfig, get_peft_model

# From the LoRA paper, we should target the q and v module.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)


from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

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

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()