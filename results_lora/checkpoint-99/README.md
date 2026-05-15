---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
---

# Checkpoint 99 - LoRA Adapter Snapshot

This directory contains the training checkpoint saved by the LoRA fine-tuning workflow in `train_model.py`.

## Directory Contents

- `adapter_config.json` — LoRA adapter configuration used during fine-tuning.
- `adapter_model.safetensors` — Adapter weights in safe tensor format.
- `tokenizer.json` — Tokenizer vocabulary and encoding JSON.
- `tokenizer_config.json` — Tokenizer configuration metadata.
- `chat_template.jinja` — Prompt template used for formatting chat input.
- `optimizer.pt` — Optimizer state checkpoint.
- `scheduler.pt` — Learning rate scheduler state checkpoint.
- `rng_state.pth` — Random number generator state for reproducibility.
- `trainer_state.json` — Trainer state and checkpoint metadata.
- `training_args.bin` — Serialized `TrainingArguments` used for the run.

## Purpose

This checkpoint stores the LoRA adapter state at step 99 of the training process. It is suitable for:

- resuming training from the same checkpoint
- exporting or inspecting adapter weights
- loading the adapter for inference with the base model

## Loading the Adapter

Use the Hugging Face `transformers` and `peft` libraries to load the base model and attach this adapter.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, ".")

tokenizer = AutoTokenizer.from_pretrained(".")
```

## Notes

- `adapter_model.safetensors` contains only the LoRA adapter weights, not the full base model.
- `training_args.bin`, `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` are primarily used for checkpoint resumption and are not required for inference.
- `chat_template.jinja` is a prompt template included for formatting conversation input.
- Ensure the base model and tokenizer are compatible with the adapter before inference.

## License

Refer to the repository-level `LICENSE` file for licensing details.
