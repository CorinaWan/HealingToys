import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import chromadb 
import logging
import asyncio
import socket
from zhconv import zhconv
import re

# ---------- Configuration logs ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI and enable automatic slash redirection.
app = FastAPI(title="RAG-Enhanced Healing Chatbot API", redirect_slashes=True)

# ---------- CORS Setting ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model and equipment configuration ----------
MODEL_PATH = "fine_tuned_deepseek_multi_turn"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
generate_lock = asyncio.Lock() # Used to prevent and cause memory overflow.

# 4-bit quantization configuration (key to solving slow operation)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Loading the language model
try:
    logger.info(f"Loading model from {MODEL_PATH} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True
    )
    logger.info("Loaded fine-tuned model.")
except Exception as e:
    logger.warning(f"Local model not found: {e}. Falling back to {FALLBACK_MODEL}.")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FALLBACK_MODEL,
        quantization_config=quantization_config if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# If there is no GPU, manually move the model to the CPU.
if device == "cpu":
    model.to(device)
model.eval()

# ---------- RAG search component ----------
retriever_model = None
chroma_collection = None

def init_rag(persist_dir: str = "./chroma_db"):
    global retriever_model, chroma_collection
    if retriever_model is None:
        logger.info("Loading SentenceTransformer for RAG...")
        retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    if chroma_collection is None:
        logger.info(f"Connecting to ChromaDB at {persist_dir}...")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("healing_toys")
    return retriever_model, chroma_collection

def retrieve_relevant_docs(query: str, top_k: int = 2):
    try:
        model_rag, collection = init_rag()
        query_emb = model_rag.encode([query]).tolist()
        results = collection.query(query_emb, n_results=top_k)
        return results
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

def detect_language_variant(text: str) -> str:
    """
    返回语言变体标识:
    - 'zh-Hant' : Traditional Chinese
    - 'zh-Hans' : Simplified Chinese
    - 'en'      : English
    """
    if not re.search(r'[\u4e00-\u9fff]', text):
        return 'en'
    converted_to_simp = zhconv.convert(text, 'zh-hans')
    if converted_to_simp == text:
        return 'zh-Hans'
    else:
        return 'zh-Hant'

def build_rag_prompt(user_query: str, retrieved_docs: dict) -> str:
    docs = retrieved_docs['documents'][0]
    metadatas = retrieved_docs['metadatas'][0]
    context_parts = []
    for doc, meta in zip(docs, metadatas):
        doc_type = meta.get('type', 'unknown')
        name = meta.get('name', meta.get('title', ''))
        context_parts.append(f"【{doc_type}】{name}\n{doc[:200]}...")
    knowledge_context = "\n\n".join(context_parts) if context_parts else "（無特定相關知識）"
    
    return f"""你是一位溫暖、貼心的心理健康助手，專門為學生提供情感支持和減壓建議。請基於以下相關知識，友好地回答用戶的問題。

【相關知識】
{knowledge_context}

用戶問題：{user_query}

請用中文或英文（根據用戶輸入語言選擇）給出有幫助、共情的回答，可以推薦合適的玩偶或提供減壓技巧。注意：如果用戶有嚴重心理危機，請引導他們尋求專業幫助。

回答："""

# ---------- Core generation logic ----------
async def get_chatbot_reply(message: str, max_tokens: int = 1024, use_rag: bool = True):
    async with generate_lock:
        context = ""
        if use_rag:
            try:
                # Keep your original RAG search logic here.
                results = retrieve_relevant_docs(message)
                context = "\n".join(results['documents'][0])
            except Exception as e:
                logger.error(f"RAG error: {e}")

        # Detecting language variations in user input
        lang_variant = detect_language_variant(message)
        
        if lang_variant == 'zh-Hant':
            lang_instruction = "你必須用繁體中文回覆，並稱呼對方為「你」，嚴禁使用「他」或「該學生」。"
        elif lang_variant == 'zh-Hans':
            lang_instruction = "你必须用简体中文回复，并称呼对方为「你」，严禁使用「他」或「该学生」。"
        else:
            lang_instruction = "You MUST respond in English and address the user as 'you'."

        # Modify the Prompt constructor to include the System Prompt.
        # This assumes you are using DeepSeek or a similar command-line fine-tuning model format.

        system_prompt = """你現在是「Healing Toys」電商平台的專業 AI 諮商師，不是 Play Even。
你對自家產品極其熟悉，你的任務是安撫學生並直接推薦玩偶。

嚴格規則：
1. 稱呼：必須使用「你」，禁止使用「他」或「該學生」。
2. 禁止反問：直接給予推薦，不要問用戶「你需要什麼類型」。
3. 產品範圍：你只能推薦以下 13 個玩偶：歡歡、安安、欣欣、琪琪、駿駿、樂樂、喜喜、米米、盈盈、詩詩、小獅、熊熊、晴晴。
4. 禁止分析：禁止輸出任何「我應該考慮」、「首先」等分析字眼。
"""
        
        full_prompt = f"<|system|>\n{system_prompt}\n\n知識庫：\n{context}\n<|user|>\n{message}\n<|assistant|>\n"

        # Below is your original tokenizer and model.generate code
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    async with generate_lock: # Ensure the generation process is thread-safe
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1, # Lower the temperature to increase stability
                top_p=0.9,
                repetition_penalty=1.2, # Slightly increase penalties to avoid repeating User/Assistant tags
                pad_token_id=tokenizer.eos_token_id
            )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    full_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
    reply = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip() # Forcefully remove the <think> tag content

    reply = reply.replace('</assistant>', '').replace('Assistant:', '').strip() # Remove labels that the model may be hallucinating

    # Remove tags or analysis beginnings that the model might be mistaking.
    # If the model response still contains words like "analysis" or "user," force truncation or replacement.
    if "分析" in reply[:50] or "首先" in reply[:50]:
        # Try to find conversational paragraphs (usually at the end)
        segments = reply.split('\n\n')
        if len(segments) > 1:
            reply = segments[-1]
            
    # 4. Make sure there are no words like "this student" (post-processing replacement)
    reply = reply.replace("這位中四學生", "你").replace("該學生", "你")
    
    return reply

# ---------- API Data Model ----------
class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    use_rag: bool = True

class ChatResponse(BaseModel):
    reply: str

# ---------- Route definition ----------

@app.post("/chat", response_model=ChatResponse)
async def chat_post(request: ChatRequest):
    try:
        reply = await get_chatbot_reply(request.message, request.max_tokens, request.use_rag)
        return ChatResponse(reply=reply)
    except (ConnectionResetError, socket.error, asyncio.CancelledError):
        raise HTTPException(status_code=499, detail="Client closed connection")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat")
async def chat_get(message: str, max_tokens: int = 1024, use_rag: bool = True):
    try:
        reply = await get_chatbot_reply(message, max_tokens, use_rag)
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"GET Request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "RAG-Enhanced Healing Chatbot API is running. Use /docs for documentation."}

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

if __name__ == "__main__":
    import uvicorn
    init_rag() # Preheating RAG Components
    uvicorn.run(app, host="0.0.0.0", port=8000)