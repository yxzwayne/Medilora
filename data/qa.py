import openai
import backoff
from tqdm import tqdm
from openai import Client
import pandas as pd

client = Client(api_key="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")
df = pd.read_csv("patient_data.csv")
tqdm.pandas()

@backoff.on_exception(backoff.constant, openai.RateLimitError)
def obtain_qa_openai(note):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a medical expert and a professional doctor focusing on the task of reconstructing the medical inquiery and doctor answer pairs from a set of given medical notes into legitimate JSON objects.",
            },
            {
                "role": "user",
                "content": f'MEDICAL NOTE: """\n{note}""" \nBased on the given medical note, construct one `Question` and `Answer` pair between the patient and the doctor in JSON format with exactly one pair of `Question` and `Answer`. The patients question includes clear and detailed description of the problem relevant in the medical note. The doctors answer is in first person perspective, and includes reasoning and details, such as sympotoms, diagnosis, inference, suggestions, medications. Both questions and answers should be concise, straight to the point and highly medically relevant. ',
            },
        ],
    )
    return response.choices[0].message.content

df['qa_pair'] = df['notes'].progress_apply(obtain_qa_openai)