from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import torch
import json
from typing import List, Dict

# [The dataset category retains your logic, but it is recommended to check max_length.]
class MultiTurnConversationDataset(Dataset):
    def __init__(self, conversations: List[List[Dict]], tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for conv in conversations:
            for i in range(len(conv)):
                if conv[i]["role"] == "assistant":
                    context = conv[:i]
                    input_text = self.format_context(context)
                    output_text = conv[i]["content"]
                    full_text = input_text + output_text
                    tokenized = tokenizer(
                        full_text,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    input_ids = tokenized["input_ids"].squeeze(0)
                    attention_mask = tokenized["attention_mask"].squeeze(0)
                    input_tokens = tokenizer(input_text, truncation=True, max_length=max_length)
                    input_len = len(input_tokens["input_ids"])
                    labels = input_ids.clone()
                    labels[:input_len] = -100
                    self.examples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })
    
    def format_context(self, context: List[Dict]) -> str:
        formatted = ""
        for msg in context:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_cleaned_conversations(jsonl_path: str) -> List[List[Dict]]:
    conversations = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            conversations.append(data["messages"])
    return conversations

def augment_with_knowledge_base(conversations):
    """Convert knowledge base content into training samples (including products, dolls, and stories)"""
    # Load all knowledge base files
    with open("knowledge_base.json", 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    with open("healing_toys_brief.json", 'r', encoding='utf-8') as f:
        brief_data = json.load(f)
    
    with open("touching_story.json", 'r', encoding='utf-8') as f:
        stories = json.load(f)
    
    kb_conversations = []
    
    # 1. Product Dialogue
    for product in kb.get("products", []):
        name = product.get("name", "")
        desc = product.get("description", "")
        short_desc = product.get("short_description", "")
        
        kb_conversations.append({
            "messages": [
                {"role": "user", "content": f"Could you please introduce {name}?"},
                {"role": "assistant", "content": f"{name}：{short_desc}\n{desc}"}
            ]
        })
        
        kb_conversations.append({
            "messages": [
                {"role": "user", "content": f"Do you have{name}？"},
                {"role": "assistant", "content": f"Yes!{name}can help you.{short_desc}"}
            ]
        })
    
    # 2. Doll Dialogue
    for doll in kb.get("dolls", []):
        name = doll.get("name", "")
        features = doll.get("features_zh", [])
        features_text = "、".join(features) if isinstance(features, list) else features
        
        kb_conversations.append({
            "messages": [
                {"role": "user", "content": f"What are the characteristics of {name}?"},
                {"role": "assistant", "content": f"{name}The doll has the following characteristics:{features_text}. It can give you warmth and companionship."}
            ]
        })
    
    # 3. Stress Reduction Techniques Dialogue
    for item in brief_data:
        if item.get("category") == "stress_relief_tips":
            title = item.get("title", "")
            tips = item.get("tips", [])
            for tip in tips[:3]:  # Each skill generates a dialogue.
                tip_name = tip.get("name_zh", tip.get("name", ""))
                tip_desc = tip.get("description_zh", tip.get("description", ""))
                
                kb_conversations.append({
                    "messages": [
                        {"role": "user", "content": f"What are some stress-relief methods?"},
                        {"role": "assistant", "content": f"You could try{tip_name}：{tip_desc}"}
                    ]
                })
    
    # 4. Inspirational Story Dialogue
    for story in stories[:5]:  # Get the first 5 stories
        title = story.get("title", "")
        theme = story.get("theme", "")
        story_text = story.get("story", "")[:200]  # Extract the first 200 words
        
        kb_conversations.append({
            "messages": [
                {"role": "user", "content": f"Can I share a story with {theme}?"},
                {"role": "assistant", "content": f"Of course!{title}：{story_text}..."}
            ]
        })
    
    print(f"{len(kb_conversations)} training samples were generated from the knowledge base")
    return conversations + kb_conversations

def train():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 1. Introducing 4-bit quantization configuration (QLoRA core: reducing memory pressure)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Loading the model in a quantized manner
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={" ": 0},
        trust_remote_code=True
    )

    # 3. Preparing for LoRA training (training only a small subset of parameters significantly improves speed)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Common modules for the Qwen architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    convs = load_cleaned_conversations("cleaned_conversations.jsonl")
    dataset = MultiTurnConversationDataset(convs, tokenizer, max_length=384) # Optimization: shorten length to speed up training

    # 4. Training parameter optimization
    training_args = TrainingArguments(
        output_dir="./results_lora",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,     # Adjust according to batch_size
        learning_rate=2e-4,                # LoRA typically requires a large learning rate.
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), # Using bf16 during support is more stable and faster.
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        optim="adamw_torch",         # Optimizer designed specifically for quantization training
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    trainer.train()
    
    # 5. Save LoRA weights (Note: This will only save the adapter weights)
    model.save_pretrained("./fine_tuned_deepseek_lora")
    tokenizer.save_pretrained("./fine_tuned_deepseek_lora")
    print("LoRA 微調完成")

if __name__ == "__main__":
    train()