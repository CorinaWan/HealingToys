import json
import re
from typing import List, Dict, Any

# System-suggested keywords (Chinese and English) to be filtered
SYSTEM_PROMPT_PATTERNS = [
    r"你是一个有用的助手[，,。]?",
    r"你是一个有用的助手，请回答用户的问题。",
    r"You are a helpful assistant[.,]?",
    r"You are a helpful assistant\. Please answer the user's question\.",
    r"请回答用户的问题",
    r"Please answer the user's question",
]

def is_system_prompt(text: str) -> bool:
    """Determine if the message content is a system prompt."""
    if not isinstance(text, str):
        return False
    for pattern in SYSTEM_PROMPT_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def clean_conversation(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Clean the messages array of a single conversation:
    - Delete messages with role='assistant' and content that are system prompts
    - Return the cleaned list of messages
    """
    cleaned = []
    for msg in messages:
        if msg.get("role") == "assistant" and is_system_prompt(msg.get("content", "")):
            continue   # Skip system prompts
        # Preserve the original message (must include role and content).
        if "role" in msg and "content" in msg and msg["content"].strip():
            cleaned.append(msg)
    return cleaned

def extract_conversations_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Extract all valid conversations from the original JSON file and return a list, where each element is {'messages': [...]}"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)   # The entire file is a list

    valid_conversations = []
    for doc in data:
        # Only process documents that contain the messages field.
        if "messages" not in doc:
            continue
        messages = doc["messages"]
        # Cleaning
        cleaned_msgs = clean_conversation(messages)
        # There must be at least two messages, and the roles at the beginning and end should be reasonable (optional: alternating between user and assistant).
        if len(cleaned_msgs) >= 2:
            # Optional: Ensure the first message is either "user" or "assistant"? In a normal conversation, the first message might be either "assistant" or "user," so keep both.
            valid_conversations.append({"messages": cleaned_msgs})
    return valid_conversations

def save_clean_conversations(conversations: List[Dict], output_jsonl: str):
    """Save as JSONL format, one conversation per line"""
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

def extract_knowledge_base(json_path: str) -> Dict[str, Any]:
    """Extract information about dolls and products, and return a dictionary"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dolls = []
    products = []
    for doc in data:
        if "dolls" in doc:
            dolls.extend(doc["dolls"])
        if "products" in doc:
            products.extend(doc["products"])
    # Remove duplicates (by name)
    unique_dolls = {d["name"]: d for d in dolls}.values()
    unique_products = {p["name"]: p for p in products}.values()
    return {
        "dolls": list(unique_dolls),
        "products": list(unique_products)
    }

if __name__ == "__main__":
    input_json = "chatbot_db.training_data.json"
    output_conversations = "cleaned_conversations.jsonl"
    knowledge_output = "knowledge_base.json"
    
    # 1. Extract and clean the dialogue
    convs = extract_conversations_from_json(input_json)
    print(f"提取到 {len(convs)} 条有效对话")
    save_clean_conversations(convs, output_conversations)
    print(f"已保存到 {output_conversations}")
    
    # 2. Extracting knowledge base
    kb = extract_knowledge_base(input_json)
    with open(knowledge_output, 'w', encoding='utf-8') as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"知识库保存到 {knowledge_output}，包含 {len(kb['dolls'])} 个玩偶，{len(kb['products'])} 个产品")