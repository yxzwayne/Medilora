from llama_cpp import Llama
import json
from tqdm import tqdm


target_fn = "eval/PubMedQA/ori_pqal.json"

# Update your model name file (gguf format) here. This can be modified with any other inference backend.
model_fn = "models/"

with open(target_fn) as f:
    data = json.load(f)

system = "As an expert doctor in clinical science and medical knowledge, can you tell me if the following question is correct, given the accompanying context? Answer 'yes', 'no', or 'maybe'. Then, follow up with some explanations."

llm = Llama(
    model_fn,
    chat_format="chatml",
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
)

results = {}

for key in tqdm(data.keys()):
    value = data[key]
    contexts = value.get("CONTEXTS", None)
    final_decision = value.get("final_decision", None)
    long_answer = value.get("LONG_ANSWER", None)
    question = value.get("QUESTION", None)

    input = " Contexts: " + " ".join(contexts) + " Question: " + question

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
    )

    # Extract the response
    response = output["choices"][0]["message"]["content"]

    # Store the results
    results[key] = {
        "CONTEXTS": contexts,
        "final_decision": final_decision,
        "LONG_ANSWER": long_answer,
        "QUESTION": question,
        "RESPONSE": response,
    }

# Write the results to a new JSON file
with open(f"pred_openhermes.json", "w") as f:
    json.dump(results, f)