from llama_cpp import Llama
import json
from tqdm import tqdm


target_fn = "test.jsonl"
# target_fn = "4_options/phrases_no_exclude_test.jsonl"

data = []

with open(target_fn, "r") as f:
    for line in f:
        data.append(json.loads(line))

# This needs to be changed based on 4 or 5 options. 
system = "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E, and then offer some step-by-step reasonings. Base your answer on the current and standard practices referenced in medical guidelines."

llm = Llama(
    "models/guideline-medqa-Q3_K_L.gguf",
    chat_format="chatml",
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
)

results = []

for entry in tqdm(data):
    question = entry["question"]
    options = entry["options"]
    answer = entry["answer"]
    answer_key = [k for k, v in options.items() if v == answer][0]
    formatted_options = "\n".join([f"({k}) {v}" for k, v in options.items()])

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\nOptions:\n{formatted_options}"},
        ]
    )
    response = output["choices"][0]["message"]["content"]
    entry["response"] = response
    results.append(entry)

# Write the results to a new JSON file
with open(f"predict_base_5option.jsonl", "w") as f:
    json.dump(results, f)