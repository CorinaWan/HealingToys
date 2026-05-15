import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any

def build_rag_knowledge_base(
    knowledge_json: str = "knowledge_base.json",
    brief_json: str = "healing_toys_brief.json",
    story_json: str = "touching_story.json",
    persist_dir: str = "./chroma_db"
):
    """Store all information about toys/products, platform introductions, stress-relief techniques, and inspirational stories into a Chroma vector database."""
    
    # 1. Load all JSON data
    with open(knowledge_json, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    
    with open(brief_json, 'r', encoding='utf-8') as f:
        brief_data = json.load(f)  # The list contains information such as platform introduction and background details for each element.
    
    with open(story_json, 'r', encoding='utf-8') as f:
        stories = json.load(f)      # The list, each element is an inspirational story.
    
    # 2. Initialize the embedding model and Chroma client
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    # Delete the old set (if it exists).
    try:
        chroma_client.delete_collection("healing_toys")
    except:
        pass
    collection = chroma_client.create_collection("healing_toys")
    
    docs = []
    metadatas = []
    ids = []
    
    # 3. Handling dolls
    for idx, doll in enumerate(kb.get("dolls", [])):
        name = doll.get("name", "")
        features_zh = doll.get("features_zh", [])
        features_en = doll.get("features_en", [])
        # Compatibility features can be a string or a list.
        if isinstance(features_zh, str):
            features_zh = [features_zh]
        if isinstance(features_en, str):
            features_en = [features_en]
        text = f"玩偶名称：{name}\n中文特征：{' '.join(features_zh)}\n英文特征：{' '.join(features_en)}"
        docs.append(text)
        metadatas.append({"type": "doll", "name": name, "source": "dolls"})
        ids.append(f"doll_{idx}")
    
    # 4. Processing products
    for idx, product in enumerate(kb.get("products", [])):
        name = product.get("name", "")
        desc = product.get("description", "")
        short_desc = product.get("short_description", "")
        text = f"产品名称：{name}\n简介：{short_desc}\n详细描述：{desc}"
        docs.append(text)
        metadatas.append({"type": "product", "name": name, "source": "products"})
        ids.append(f"product_{idx}")
    
    # 5. Process the various items in healing_toys_brief
    for idx, item in enumerate(brief_data):
        category = item.get("category", "")
        title = item.get("title", "")
        
        if category == "platform_intro":
            content = item.get("content", "")
            text = f"平台介绍 - {title}\n{content}"
        elif category == "background":
            content_en = item.get("content_en", "")
            content_zh = item.get("content_zh", "")
            mission_en = item.get("mission_en", "")
            mission_zh = item.get("mission_zh", "")
            text = f"背景与使命\n英文背景：{content_en}\n中文背景：{content_zh}\n英文使命：{mission_en}\n中文使命：{mission_zh}"
        elif category == "stress_relief_tips":
            tips = item.get("tips", [])
            tips_text = ""
            for tip in tips:
                name = tip.get("name", "")
                name_zh = tip.get("name_zh", "")
                desc = tip.get("description", "")
                desc_zh = tip.get("description_zh", "")
                tips_text += f"技巧：{name}（{name_zh}）\n说明：{desc}\n中文说明：{desc_zh}\n\n"
            text = f"减压技巧 - {title}\n{tips_text}"
        elif category == "ai_chatroom":
            desc = item.get("description", "")
            desc_zh = item.get("description_zh", "")
            text = f"AI 聊天室 - {title}\n英文介绍：{desc}\n中文介绍：{desc_zh}"
        else:
            continue  # Skip Unknown Type
        
        docs.append(text)
        metadatas.append({"type": "brief", "category": category, "title": title})
        ids.append(f"brief_{idx}")
    
    # 6. Handling inspirational stories
    for idx, story in enumerate(stories):
        title = story.get("title", "")
        theme = story.get("theme", "")
        story_text = story.get("story", "")
        text = f"励志故事标题：{title}\n主题：{theme}\n故事内容：{story_text}"
        docs.append(text)
        metadatas.append({"type": "story", "title": title, "theme": theme})
        ids.append(f"story_{idx}")
    
    # 7. Calculate embeddings and store them in Chroma.
    print(f"正在为 {len(docs)} 条记录生成向量...")
    embeddings = model.encode(docs).tolist()
    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"知识库构建完成！共 {len(docs)} 条记录（玩偶: {len(kb.get('dolls',[]))}, 产品: {len(kb.get('products',[]))}, 平台内容: {len(brief_data)}, 故事: {len(stories)}）")


def retrieve_relevant_docs(query: str, top_k: int = 3, persist_dir: str = "./chroma_db"):
    """检索与用户问题最相关的文档（玩偶/产品/故事/平台内容等）"""
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_collection("healing_toys")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_emb = model.encode([query]).tolist()
    results = collection.query(query_emb, n_results=top_k)
    return results


if __name__ == "__main__":
    # Build a complete knowledge base (ensure the three JSON files are in the same directory).
    build_rag_knowledge_base()
    
    # Test retrieval
    test_query = "我压力很大，需要一个温柔的玩偶陪我"
    results = retrieve_relevant_docs(test_query)
    print("\n检索结果：")
    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"\n--- 结果 {i+1} (距离: {dist:.4f}) ---")
        print(f"类型: {meta.get('type')}")
        print(f"元数据: {meta}")
        print(f"内容预览: {doc[:200]}...")