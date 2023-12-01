import torch
import random
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv("patient_data.csv")
sample_note = df.iloc[random.randint(0, len(df.index))].notes

model_id = "mlabonne/NeuralHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    tokenizer=tokenizer, 
    # torch_dtype=torch.bfloat16, 
    device_map="auto"
)


def qa_acc(note):
    # Format prompt
    message = [
        {
            "role": "system",
            "content": "You are a medical expert and a professional doctor focusing on the task of reconstructing the medical inquiery and doctor answer pairs from a set of given medical notes into legitimate JSON objects.",
        },
        {
            "role": "user",
            "content": f'MEDICAL NOTE: """\n{note}""" \nBased on the given medical note, construct one `Question` and `Answer` pair between the patient and the doctor in JSON format with exactly one pair of `Question` and `Answer`. The patients question includes clear and detailed description of the problem relevant in the medical note. The doctors answer is in first person perspective, and includes reasoning and details, such as sympotoms, diagnosis, inference, suggestions, medications. Both questions and answers should be concise, straight to the point and highly medically relevant. ',
        },
    ]
    prompt = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False
    )

    # Generate text
    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=1,
        max_length=2834
    )
    print(sequences[0]["generated_text"])


qa_acc(sample_note)