from datasets import load_dataset, Dataset, DatasetDict


dataset = load_dataset("pubmed_qa", "pqa_artificial")["train"]
dataset = dataset.remove_columns("pubid")
dataset = dataset.flatten().remove_columns(["context.labels", "context.meshes"])

dataset = dataset.rename_column(
    original_column_name="context.contexts", new_column_name="contexts"
)


def format_message(role: str, message: str) -> str:
    return f"<|im_start|>{role}\n{message}<|im_end|>\n"


print("Finished loading the dataset, starting processing")


def formatting_prompts_func(row):
    system_prompt = "As an expert doctor in clinical science and medical knowledge, can you tell me if the following question is correct, given the accompanying context? Answer yes, no, or maybe. Then, follow up with some explanations."
    context = " ".join(row["contexts"])
    question = row["question"]
    final_decision = row["final_decision"]
    long_answer = row["long_answer"]

    # Combine context and question for the question
    question_text = "Context: " + context + " Question: " + question
    # Combine final_decision and long_answer for the answer
    answer_text = final_decision + ". " + long_answer

    # Format the messages
    system = format_message("system", system_prompt)
    question = format_message("user", question_text)
    answer = format_message("assistant", answer_text)

    # Combine all messages
    text = system + question + answer

    # Create a dictionary for each example
    example_dict = {
        "system_prompt": system_prompt,
        "question": question_text,
        "response": answer_text,
    }
    # output.append(example_dict)


# Assuming the output_text is a list of dictionaries with keys 'system_prompt', 'question', 'response'
output_text = formatting_prompts_func(dataset)

# Convert the list of dictionaries to a Dataset
formatted_dataset = Dataset.from_dict(
    {k: [dic[k] for dic in output_text] for k in output_text[0]}
)

# Create a DatasetDict
dataset_dict = DatasetDict({"train": formatted_dataset})

# Upload the dataset to Hugging Face
dataset_dict.upload_to_hub("Medilora/PubMed-OpenOrca-Instruct")
