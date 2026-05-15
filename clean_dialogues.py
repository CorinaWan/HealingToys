#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Healing Toys 对话数据清洗脚本
从混合 JSON 文件中提取多轮对话，输出 JSONL 格式
"""

import json
import re
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional

# ---------- 配置 ----------
INPUT_JSON_PATH = "chatbot_db.training_data.json"   # 原始 JSON 文件
OUTPUT_JSONL_PATH = "cleaned_conversations.jsonl"  # 输出文件
MIN_RESPONSE_LEN = 10                               # 最短回复长度
INVALID_PHRASES = [
    "I'm really sorry", "我是一个人工智能", "作为AI",
    "你是一个有用的助手", "You are a helpful assistant"
]   # 包含这些短语的回答将被过滤（可根据需要调整）

# ---------- 辅助函数 ----------
def clean_text(text: str) -> str:
    """去除多余空白和特殊字符，但保留中英文标点"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # 移除多余的换行和制表符，但保留句号等
    text = re.sub(r'\s+', ' ', text)
    return text

def is_valid_response(text: str) -> bool:
    """判断回复是否有效（长度足够，不是占位符）"""
    if len(text) < MIN_RESPONSE_LEN:
        return False
    # 检查无效短语
    for phrase in INVALID_PHRASES:
        if phrase.lower() in text.lower():
            return False
    # 如果全是标点或数字，无效
    if re.fullmatch(r'[\d\s\.,!?;:""''()\[\]{}]+', text):
        return False
    return True

def extract_messages_conversation(doc: Dict) -> Optional[List[Dict]]:
    """
    从标准 messages 数组中提取多轮对话
    返回 [{"role": "user", "content": "..."}, ...] 或 None
    """
    messages = doc.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    # 统一转换为标准格式，过滤 role 不是 user/assistant 的消息
    conv = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant"):
            continue
        if not content or not isinstance(content, str):
            continue
        content = clean_text(content)
        if not content:
            continue
        conv.append({"role": role, "content": content})

    # 必须至少有一对 user-assistant，且最后一条是 assistant
    if len(conv) < 2:
        return None
    # 如果以 assistant 开头，去掉开头的 assistant（避免不完整）
    while conv and conv[0]["role"] == "assistant":
        conv.pop(0)
    if len(conv) < 2:
        return None
    # 确保最后一个 role 是 assistant
    if conv[-1]["role"] != "assistant":
        return None
    return conv

def extract_text_output_pair(doc: Dict) -> Optional[Tuple[str, str]]:
    """从 text + output 字段提取单轮对话 (user, assistant)"""
    user_text = doc.get("text")
    assistant_text = doc.get("output")
    if not user_text or not assistant_text:
        return None
    user_text = clean_text(user_text)
    assistant_text = clean_text(assistant_text)
    # 移除可能的前缀 "你是一个有用的助手，请回答用户的问题："
    user_text = re.sub(r'^你是一个有用的助手，请回答用户的问题：', '', user_text)
    user_text = re.sub(r'^You are a helpful assistant. Please answer the user\'s question: ', '', user_text)
    if user_text and assistant_text and is_valid_response(assistant_text):
        return (user_text, assistant_text)
    return None

def extract_generations_pair(doc: Dict) -> Optional[Tuple[str, str]]:
    """
    处理 generations 结构（常见于 LangChain LLMResult）
    通常文档中还有 text 字段作为用户输入
    """
    generations = doc.get("generations")
    if not isinstance(generations, list) or not generations:
        return None
    # 尝试获取第一个 generation 的 text
    first_gen = generations[0]
    if isinstance(first_gen, list) and len(first_gen) > 0:
        assistant_text = first_gen[0].get("text")
    elif isinstance(first_gen, dict):
        assistant_text = first_gen.get("text")
    else:
        return None
    if not assistant_text:
        return None
    assistant_text = clean_text(assistant_text)
    if not is_valid_response(assistant_text):
        return None
    # 用户输入通常在同一文档的 text 字段
    user_text = doc.get("text")
    if user_text:
        user_text = clean_text(user_text)
        user_text = re.sub(r'^你是一个有用的助手，请回答用户的问题：', '', user_text)
        return (user_text, assistant_text)
    return None

def extract_response_zh_en_pair(doc: Dict) -> Optional[Tuple[str, str]]:
    """处理 response_zh / response_en 字段，构建指令-回复对"""
    if "response_zh" in doc and isinstance(doc["response_zh"], dict):
        msg = doc["response_zh"].get("message")
        if msg and is_valid_response(msg):
            # 自动构造一个问题（可根据上下文优化）
            question = "请给我一些压力管理的建议。"
            return (question, clean_text(msg))
    if "response_en" in doc and isinstance(doc["response_en"], dict):
        msg = doc["response_en"].get("message")
        if msg and is_valid_response(msg):
            question = "Please give me some stress management advice."
            return (question, clean_text(msg))
    return None

def extract_dolls_knowledge(doc: Dict) -> List[Tuple[str, str]]:
    """从 dolls 数组生成知识问答对"""
    pairs = []
    dolls = doc.get("dolls")
    if not isinstance(dolls, list):
        return pairs
    for doll in dolls:
        name = doll.get("name", "")
        features_zh = doll.get("features_zh", [])
        features_en = doll.get("features_en", [])
        # 中文介绍
        if features_zh:
            description = " ".join(features_zh)
            question_zh = f"介绍一下{name}玩偶的特点。"
            answer_zh = f"{name}是一个{description}"
            if len(answer_zh) >= MIN_RESPONSE_LEN:
                pairs.append((question_zh, clean_text(answer_zh)))
        # 英文介绍
        if features_en:
            description_en = " ".join(features_en)
            question_en = f"Introduce the features of the {name} doll."
            answer_en = f"{name} is a {description_en}"
            if len(answer_en) >= MIN_RESPONSE_LEN:
                pairs.append((question_en, clean_text(answer_en)))
    return pairs

def extract_ai_counsellor_intro(doc: Dict) -> Optional[Tuple[str, str]]:
    """处理 ai_counsellor_intro 字段，生成自我介绍对话"""
    intro = doc.get("ai_counsellor_intro")
    if intro and isinstance(intro, str):
        question = "你是谁？你能做什么？"
        answer = clean_text(intro)
        if is_valid_response(answer):
            return (question, answer)
    return None

def single_turn_to_conversation(user: str, assistant: str) -> List[Dict]:
    """将单轮 (user, assistant) 转换为多轮对话格式"""
    return [
        {"role": "user", "content": clean_text(user)},
        {"role": "assistant", "content": clean_text(assistant)}
    ]

def deduplicate_conversations(convs: List[List[Dict]]) -> List[List[Dict]]:
    """基于 JSON 字符串去重对话列表"""
    seen = set()
    unique = []
    for conv in convs:
        key = json.dumps(conv, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(conv)
    return unique

# ---------- 主处理流程 ----------
def main():
    print(f"读取 JSON 文件: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误：顶层 JSON 不是数组")
        return

    all_conversations = []   # 存储多轮对话列表

    for idx, doc in enumerate(data):
        doc_id = doc.get("_id", f"doc_{idx}")
        # 1. 处理标准 messages 数组
        conv = extract_messages_conversation(doc)
        if conv:
            all_conversations.append(conv)
            continue   # 如果已有完整多轮，不再生成其他单轮，避免重复

        # 2. 处理 text + output
        pair = extract_text_output_pair(doc)
        if pair:
            all_conversations.append(single_turn_to_conversation(*pair))

        # 3. 处理 generations
        pair = extract_generations_pair(doc)
        if pair:
            all_conversations.append(single_turn_to_conversation(*pair))

        # 4. 处理 response_zh/en
        pair = extract_response_zh_en_pair(doc)
        if pair:
            all_conversations.append(single_turn_to_conversation(*pair))

        # 5. 处理 dolls 知识
        doll_pairs = extract_dolls_knowledge(doc)
        for user, assistant in doll_pairs:
            all_conversations.append(single_turn_to_conversation(user, assistant))

        # 6. 处理 ai_counsellor_intro
        pair = extract_ai_counsellor_intro(doc)
        if pair:
            all_conversations.append(single_turn_to_conversation(*pair))

    # 去重
    print(f"去重前对话数: {len(all_conversations)}")
    all_conversations = deduplicate_conversations(all_conversations)
    print(f"去重后对话数: {len(all_conversations)}")

    # 写入 JSONL
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            # 每条对话保存为 {"conversations": [...]}
            record = {"conversations": conv}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 清洗完成！输出文件: {OUTPUT_JSONL_PATH}")

if __name__ == "__main__":
    main()