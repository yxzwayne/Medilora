import requests
from pprint import pprint
import json
import pandas as pd
import random
from tqdm import tqdm

tqdm.pandas()
df = pd.read_csv("patient_data.csv")
n = len(df.index)
df.head()


def ollama_request(prompt, model="hermes"):
    url = "http://localhost:11434/api/generate"
    param = {"model": "hermes", "prompt": prompt, "stream": False, "raw": True}
    res = requests.post(url, json=param).json()
    bot_response = res["response"]
    if len(bot_response.strip()) != 0:
        sec = res["eval_duration"] / 1000000000
        tok_s = res["eval_count"] / sec
        print(f"tok/s: {tok_s}")
        return bot_response
    else:
        return ""


def obtain_qa(note, model="hermes"):
    q_prompt = f'MEDICAL NOTE: """\n{note}""" \nBased on the given medical note, what is be the single most probably inquiry or question the patient asked to the doctor? '
    q = ollama_request(q_prompt, model)

    ans_prompt = f'PATIENT QUESTION:  """\n{q}""" \n MEDICAL NOTE: """\n{note}""" \nBased on the given medical note and patient question, construct a concise and terse paragraph of a top professional doctors response in 3 to 4 sentences. '
    ans = ollama_request(ans_prompt)

    return q, ans


def obtain_qa_single_run(note, model="neural"):
    prompt = f'You are a medical expert and a professional doctor focusing on the task of reconstructing the medical inquiery and doctor answer pairs from a set of given medical notes into legitimate JSON objects. MEDICAL NOTE: """\n{note}""" \n Based on the given medical note, output one `Question` and `Answer` pair between the patient and the doctor in JSON format with exactly one pair of `Question` and `Answer`. The patients question includes clear and detailed description of the problem relevant in the medical note. The doctors answer is in first person perspective, and includes reasoning and details, such as sympotoms, diagnosis, inference, suggestions, medications. Both questions and answers should be concise, straight to the point and highly medically relevant. Only write the json string and nothing else.'

    res = ollama_request(prompt, model)
    return res


# sample_note = df.iloc[random.randint(0, n)].notes

# q, ans = obtain_qa(sample_note, "neural")
# print(f"==== question: {q}")
# print(f"==== answer: {ans}")

# print(obtain_qa_single_run(sample_note, "neural"))

df["qa_pair"] = df["notes"].progress_apply(obtain_qa_single_run)

df.to_csv("mimic_synthetic_qa.csv")