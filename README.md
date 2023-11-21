# MediLora

Finetuning LLMs on medical text data to elicit differential diagnosis.

## Logging the runs on wandb

run the following lines in CLI to initiate your wandb session on the machine:

```wandb login```

## Related Work

### [ChatDoctor](https://arxiv.org/pdf/2303.14070.pdf)

Their training set up for healthCareMagic-100k was 3 hours on 6 A100 GPUs.

Training parameters:
- batch size 192
- learning rate 2e-5
- epoch 3
- max seq length 512
- warmup 0.03
- weight decay none

They constructed databases using MedlinePlus and Wikipedia as external knowledge base for inference retrieval. The construction framework of the knowledge base was claimed to be extendable to reliable online databases, such as reputable academic journals.

Format of External Database:

Disease: ...
Symptoms: ...
Further Test: ...
Treatment: ...

Their evaluation was using contemporary medical queries containing recent medical news. 

### [BioGPT](https://academic.oup.com/bib/article/23/6/bbac409/6713511)

Classical literature. Uses GPT-2 architecture on various medical NLP tasks, not limited to QA.


## Datasets

- [HealthCareMagic](https://huggingface.co/datasets/RafaelMPereira/HealthCareMagic-100k-Chat-Format-en) from the ChatDoctor paper
    - only one column of string in the format of `<human>: ... <bot>: ...`
- [PubMedQA](https://huggingface.co/datasets/pubmed_qa)
    - Columns: `pubid`, `question`, `context` (sequence), `long_answer`, `final_decision`

[PubMedQA repo](https://pubmedqa.github.io/) offers evaluation scripts. Will check repo and run benchmark against top 5 models.



## Model Checkpoints Considered

- teknium/OpenHermes-2.5-Mistral-7B
- NousResearch/Nous-Hermes-llama-2-7b
- NousResearch/Nous-Hermes-Llama2-13b