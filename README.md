# MediLora

Langauge model fine-tuning with limited datasets and Q-LoRA in data-limited scenarios.

[Code](https://github.com/yxzwayne/Medilora) | [Report](/report/Medilora_Final_Report.pdf)


## References notes

### Steps to take on a new instance

Because we use axolotl, we don't need to use the requirements. It comes with wandb

- ~~`pip install -r requirements` (for logging the runs on wandb)~~
- `wandb login`
- `huggingface-cli login` (Connect HuggingFace for Data and Model sharing)
- If using notebook, `pip install huggingface_hub`, then run the following in a cell:

```
from huggingface_hub import notebook_login
notebook_login()
```

### Using Axolotl
For using axolotl, refer to their official documentation, but here is the commands we use:

- `accelerate launch -m axolotl.cli.train qlora.yml --deepspeed deepspeed/zero3.json` (if there's a problem, get rid of the --deepspeed flag as well.)

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


