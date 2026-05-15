---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
---

# Fine-Tuned DeepSeek LoRA Adapter

This directory contains the fine-tuned LoRA adapter artifacts produced by the repository's training workflow. It can be used to attach a lightweight adapter to the base Qwen model for inference.

## Files Included

- `adapter_config.json` — LoRA configuration containing adapter hyperparameters and target modules.
- `adapter_model.safetensors` — Saved LoRA adapter weights in safe tensor format.
- `tokenizer.json` — Tokenizer vocabulary and encoding metadata.
- `tokenizer_config.json` — Tokenizer configuration settings.
- `chat_template.jinja` — Chat prompt template used to format model input for generation.

## Purpose

This directory holds the adapter-only model artifacts, not the full base model. Use it when you want to:

- perform inference with the fine-tuned adapter attached to the base model
- deploy a lighter-weight fine-tuned variant
- inspect adapter configuration and prompt template usage

## How to Use

Load the base model and merge this adapter using `transformers` and `peft`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_deepseek_lora")
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./fine_tuned_deepseek_lora")

# Use the prompt template if needed
with open("./fine_tuned_deepseek_lora/chat_template.jinja", "r", encoding="utf-8") as f:
    template = f.read()
```

## Notes

- `adapter_model.safetensors` only contains the LoRA adapter weights and requires the original base model for full inference.
- `tokenizer.json` and `tokenizer_config.json` are required for consistent text encoding and decoding.
- `chat_template.jinja` is included to help format prompts for the fine-tuned chatbot behavior.
- If you need a complete model checkpoint including optimizer state and training progress, see `results_lora/checkpoint-99`.

## License

See the repository-level `LICENSE` file for licensing details.
