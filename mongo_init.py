from pymongo import MongoClient

def init_db():
    client = MongoClient(
        "mongodb+srv://mongodb:strongPassword@vector-search.mkmihsf.mongodb.net/?appName=Vector-Search"
    )
    db = client["chatbot_db"]
    
    # 创建集合（若不存在）
    collections = ["training_data", "conversation_history", "model_versions"]
    for name in collections:
        if name not in db.list_collection_names():
            db.create_collection(name)
    print("MongoDB initialized with collections:", collections)

if __name__ == "__main__":
    init_db()
