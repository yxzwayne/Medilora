# MediLora

Langauge model fine-tuning with limited datasets and Q-LoRA in data-limited scenarios.

[Code](https://github.com/yxzwayne/Medilora) | [Report](/report/Medilora_Final_Report.pdf)


### Model Checkpoints Considered

- From Nous Research:

  - teknium/OpenHermes-2.5-Mistral-7B
  - NousResearch/Nous-Hermes-llama-2-7b
  - NousResearch/Nous-Hermes-Llama2-13b

- From Microsoft:
  - microsoft/Orca-2-7b
  - microsoft/Orca-2-13b

**Notes**

1. Mistral-7b on OpenHermes benched higher than on llama2-13b, which makes sense because mistral is a llama2 finetune.
2. Orca-2-13b underperforms OpenHermes-2.5-Mistral-7B on BigBench, which is quite a reflection on its reasoning abilities.

### Repo Referenced

- https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/README.md


