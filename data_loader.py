from pymongo import MongoClient
from transformers import AutoTokenizer
import torch
from bson import ObjectId  # Add ObjectId type reference

def load_training_data(collection_name, limit=1000, strict=True):
    client = MongoClient("mongodb+srv://mongodb:strongPassword@vector-search.mkmihsf.mongodb.net/?appName=Vector-Search")
    db = client["chatbot_db"]
    collection = db[collection_name]
    
    data = []
    for doc in collection.find().limit(limit):
        messages = doc.get("messages", [])
        doc_id = doc.get('_id', 'unknown')
        
        # Strict mode verification
        if strict:
            if len(messages) < 2:
                print(f"⚠️ 严格模式跳过不完整对话 (ID: {str(doc_id)[:6]}...)")
                continue
                
            if not all("content" in m for m in messages[:2]):
                print(f"⚠️ 严格模式跳过内容缺失对话 (ID: {str(doc_id)[:6]}...)")
                continue
                
        # Extracting valid dialogue pairs
        for i in range(len(messages) - 1):
            user_input = messages[i].get("content")
            assistant_response = messages[i+1].get("content")
            
            if user_input and assistant_response:
                data.append((user_input, assistant_response))
            else:
                print(f"⚠️ 跳过空内容对话 (起始位置: {i}, ID: {str(doc_id)[:6]}...)")
                
    return data

def preprocess_data(data, tokenizer, max_length=512):
    """返回 input_ids, attention_mask, labels 三个张量"""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for idx, (user_input, assistant_response) in enumerate(data):
        prompt = f"User: {user_input}\nAssistant: {assistant_response}"
        encoded = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create training labels
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return (
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.stack(labels_list)
    )

if __name__ == "__main__":
    # Database connection configuration
    client = MongoClient("mongodb+srv://mongodb:strongPassword@vector-search.mkmihsf.mongodb.net/?appName=Vector-Search")
    db = client["chatbot_db"]
    collection = db["training_data"]

    # Test dataset
    test_data = [
        {
            "messages": [
                {"role": "assistant", "content": "你好！親愛的學生，今天過得怎麼樣？"},
                {"role": "user", "content": "您好！我是一名四年級的資訊與通訊科技專業的學生。最近我需要準備很多科目的考試，所以我感到壓力很大。"},
                {"role": "assistant", "content": "同學你好！聽到你正在為多科考試感到壓力，這確實是許多大學生都會經歷的階段。選擇一個能陪伴你、給你支持的玩偶是個很棒的主意，它能成為你學習時的情感小夥伴。"}
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "你好！親愛的學生，今天心情如何？"},
                {"role": "user", "content": "您好！我是一名中四學生。最近我英文科考試成績不理想，感到很沮喪，我不想再活了。"},
                {"role": "assistant", "content": "考試成績不理想不代表你不夠好，更不代表未來沒有希望。你願意主動尋找陪伴和支持的方式，這本身已經非常勇敢和值得肯定。關於玩偶的選擇，根據你提到的“感到沮喪、壓力大、需要情感支持”，我會推薦："}
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "你好！親愛的學生，今天心情如何？"},
                {"role": "user", "content": "我是中五學生，現在要應付將來的DSE文憑試，我感到很大壓力，缺乏自信心。您會推薦那一個玩偶讓我買來陪伴我嗎？"},
                {"role": "assistant", "content": "面對DSE文憑試的壓力是很正常的。選擇一個可愛的玩偶作為你的陪伴是個不錯的主意！我推薦你可以考慮購買一隻小獅，它很勇敢，願意面對各種挑戰，它能為你帶來安慰和陪伴感，從而成為你學習路上的好朋友！"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "I'm a Form 5 student, and I'm feeling a lot of pressure and lack confidence as I prepare for the upcoming DSE (Diploma of Secondary Education Examination) exams. Would you recommend a plush toy to keep me company?"},
                {"role": "assistant", "content": "Hello! Dear student, it's normal to feel pressured about the DSE exams. Choosing a cute plush toy as your companion is a great idea! I recommend considering a little lion. It's brave and willing to face all challenges, providing comfort and companionship, making it a good friend on your learning journey!"}
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "Hello! Dear student, how are you doing today?"},
                {"role": "user", "content": "Hello! I am a year 4 ICT student. Recently, I needed to prepare for examinations in many subjects, so I feel stressed out."}
            ]
        },
        {
            "messages": [
                {"role": "assistant", "content": "Hello! Dear student, how are you doing today?"},
                {"role": "user", "content": "Hello! I am a Form 4 student. I recently did poorly on my English exam and I feel very frustrated. I don't want to live anymore."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "您好！我想知道你們網購平台有什麼服務及特色。"},
                {"role": "assistant", "content": "您好！歡迎了解Healing Toys心靈療癒網購平台。我們是一個專注於青少年心理健康與情緒陪伴的線上平台，致力於透過可愛療癒玩偶與AI心理陪伴服務，幫助6-26歲的學生群體緩解學業壓力、提升情緒幸福感。"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello! I would like to know what services and features your online shopping platform offers."},
                {"role": "assistant", "content": "Hello! Welcome to Healing Toys, an online shopping platform focused on mental health and emotional support for teenagers. We are dedicated to helping students aged 6-26 alleviate academic stress and improve their emotional well-being through adorable therapeutic toys and AI-powered psychological companionship services."}
            ]
        }
    ]

    # Perform data insertion
    result = collection.insert_many(test_data)
    print(f"✅ 成功插入 {len(result.inserted_ids)} 条测试数据")
    
    # Verify data integrity
    print(f"📊 集合文档总数: {collection.count_documents({})}")
    for doc in collection.find().limit(5):
        doc_id = doc.get('_id', 'unknown')
        messages = doc.get("messages", [])
        # Fix ObjectId slicing issue
        print(f"\n文档ID: {str(doc_id)[:6]}...")
        print(f"消息数量: {len(messages)}")
        print(f"首条消息: {messages[0]['content'][:50]}...")
