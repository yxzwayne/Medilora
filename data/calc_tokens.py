from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import reduce

dataset_id = "epfl-llm/guidelines"
model_id = "Medilora/guideline-medilora-adapter"

tokenizer = AutoTokenizer.from_pretrained(model_id)


def count_tokens(batch):
    tokens = tokenizer.batch_encode_plus(batch, padding=True)["input_ids"]
    token_counts = [len(t) for t in tokens]
    return sum(token_counts)

def main():
    dataset = load_dataset(dataset_id)
    texts = dataset["train"]["clean_text"]

    batch_size = 100  # adjust based on your hardware

    with Pool() as pool:
        token_counts = list(tqdm(pool.imap(count_tokens, [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]), total=len(texts)))

    n_tokens = sum(token_counts)

    print(f"Total tokens = {n_tokens}")


if __name__ == "__main__":
    main()
