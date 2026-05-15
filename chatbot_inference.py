import logging
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Configuration logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable: RAG component (initialized only once)
retriever_model = None
chroma_collection = None

def init_rag(persist_dir: str = "./chroma_db"):
    """初始化 RAG 检索组件（SentenceTransformer + Chroma）"""
    global retriever_model, chroma_collection
    if retriever_model is None:
        logger.info("Loading SentenceTransformer model for RAG...")
        retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    if chroma_collection is None:
        logger.info(f"Connecting to ChromaDB at {persist_dir}...")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("healing_toys")
    return retriever_model, chroma_collection

def retrieve_relevant_docs(query: str, top_k: int = 3):
    """检索与用户问题最相关的文档（玩偶/产品/故事/平台内容等）"""
    try:
        model_rag, collection = init_rag()
        query_emb = model_rag.encode([query]).tolist()
        results = collection.query(query_emb, n_results=top_k)
        return results
    except Exception as e:
        logger.warning(f"RAG 檢索失敗: {e}")
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

def build_rag_prompt(user_query: str, retrieved_docs: dict) -> str:
    """
    根据检索结果构造增强 prompt
    retrieved_docs 格式与 chromadb 返回一致：
    {
        'documents': [[doc1, doc2, ...]],
        'metadatas': [[meta1, meta2, ...]],
        'distances': [[dist1, dist2, ...]]
    }
    """
    docs = retrieved_docs['documents'][0]
    metadatas = retrieved_docs['metadatas'][0]
    
    # Building knowledge context
    context_parts = []
    for idx, (doc, meta) in enumerate(zip(docs, metadatas)):
        doc_type = meta.get('type', 'unknown')
        name = meta.get('name', meta.get('title', ''))
        context_parts.append(f"【{doc_type}】{name}\n{doc[:200]}...")  # 截断过长内容
    
    knowledge_context = "\n\n".join(context_parts)
    
    # Construct a complete prompt
    prompt = f"""你是一位温暖、贴心的心理健康助手，专门为学生提供情感支持和减压建议。请基于以下相关知识，友好地回答用户的问题。

【相关知识】
{knowledge_context}

用户问题：{user_query}

请用中文或英文（根据用户输入语言选择）给出有帮助、共情的回答，可以推荐合适的玩偶或提供减压技巧。注意：如果用户有严重心理危机，请引导他们寻求专业帮助。

回答："""

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """根据 prompt 生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # Move to the device where the model is located
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode only the newly generated part
    input_len = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()

def load_optimized_model():
    """載入優化後的 LoRA 模型"""
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    lora_weights_path = "./fine_tuned_deepseek_lora" # This points to the path where you saved train_model.py.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    logger.info("正在載入基礎模型與 LoRA 權重...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map={" ": 0} if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Load LoRA adapter
    try:
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        logger.info("成功掛載 LoRA 適配器。")
    except Exception as e:
        logger.warning(f"未能載入 LoRA 權重 ({e})，將使用原始模型進行推論。")
        model = base_model
        
    model.eval()
    return model, tokenizer

def main():
    # 1. Load the optimized model
    model, tokenizer = load_optimized_model()

    # 2. Test Dialogue Example
    test_cases = [
        "我是中五學生，現在要應付將來的DSE文憑試，我感到很大壓力，缺乏自信心。",
        "我想知道你們 Healing Toys 平台有什麼特色？",
        "我最近好大壓力，有冇邊個公仔可以陪我溫書？"
    ]

    for user_input in test_cases:
        logger.info(f"處理問題: {user_input}")

        try:
            # Perform RAG search
            retrieved = retrieve_relevant_docs(user_input, top_k=2)

            # Check if the search results exist; if not, use a simple Prompt.
            if retrieved and 'documents' in retrieved and len(retrieved['documents'][0]) > 0:
                prompt = build_rag_prompt(user_input, retrieved)
            else:
                logger.info("未找到相關文檔，使用基礎 Prompt。")
                prompt = f"User: {user_input}\nAssistant:"

        except Exception as e:
            logger.error(f"構建 Prompt 出錯: {e}")
            prompt = f"User: {user_input}\nAssistant:"

        # Final defense check: Ensure prompt is not empty.
        if not prompt:
            prompt = f"User: {user_input}\nAssistant:"
        # Generate answer
        response = generate_response(model, tokenizer, prompt)
        
        print(f"\nUser: {user_input}")
        print(f"AI: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()