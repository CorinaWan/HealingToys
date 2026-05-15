# HealingToys

## Overview

`HealingToys` is a Python-based RAG-enhanced chatbot project focused on emotional support and product recommendation for students. It combines fine-tuned language model inference, knowledge-base retrieval with ChromaDB, training data preparation, and API serving through FastAPI.

## Key Components

- `chat_api.py` - FastAPI server for chatbot inference with optional RAG retrieval.
- `chatbot_inference.py` - Standalone inference workflow with optimized model loading and prompt generation.
- `build_rag_kb.py` - Builds a ChromaDB vector store from product, doll, story, and platform knowledge files.
- `train_model.py` - QLoRA training pipeline for fine-tuning a causal LM using conversation data.
- `data_loader.py` - MongoDB training data ingestion and preprocessing utilities.
- `clean_data.py` - Generic conversation cleanup tools for JSON datasets.
- `clean_dialogues.py` - Dialogue extraction and deduplication logic for cleaned data.
- `mongo_init.py` - MongoDB initialization helper.
- `healingtoys.py` - Example OpenAI chat script for external API usage.

## Data Sources

The repository includes multiple knowledge and training data files:

- `knowledge_base.json`
- `healing_toys_brief.json`
- `touching_story.json`
- `cleaned_conversations.jsonl`
- `training_data.json`
- `wc_products_data.json`
- `chatbot_db.training_data.json`
- `chatbot_db.conversation_history.json`

## Setup

### Python Dependencies

Install required Python packages in your virtual environment:

```bash
python3 -m pip install -r requirements.txt
```

> If there is no `requirements.txt`, install the main runtime packages manually:
>
> ```bash
> python3 -m pip install fastapi uvicorn transformers sentence-transformers chromadb torch peft zhconv pymongo openai
> ```

### Node Dependency

A minimal `package.json` is present with OpenAI dependency. Install it if you need the Node.js example:

```bash
npm install
```

## Running the API

Start the FastAPI service:

```bash
python3 chat_api.py
```

Then call the API at:

- `POST /chat`
- `GET /chat`
- `GET /health`
- `GET /`

The service loads a fine-tuned local model if available, otherwise falls back to a remote model.

## Building the RAG Knowledge Base

Generate the ChromaDB persistence store from local JSON content:

```bash
python3 build_rag_kb.py
```

This creates or refreshes the `./chroma_db` vector database used by retrieval.

## Training the Model

Train the model using the fine-tuning pipeline:

```bash
python3 train_model.py
```

This script:

- loads cleaned conversations from `cleaned_conversations.jsonl`
- prepares a `MultiTurnConversationDataset`
- fine-tunes with LoRA and quantization
- saves results into `./results_lora` and `./fine_tuned_deepseek_lora`

## Inference

Run standalone inference and RAG prompt generation:

```bash
python3 chatbot_inference.py
```

The script loads the optimized model, retrieves relevant documents from ChromaDB, constructs a RAG prompt, and generates sample responses.

## Data Preparation

- `clean_data.py` is used to clean raw JSON conversation data.
- `clean_dialogues.py` extracts valid user-assistant pairs and deduplicates conversations.
- `data_loader.py` can load training examples from MongoDB and preprocess them for model training.

## Notes

- `healingtoys.py` is a simple OpenAI API sample and is separate from the main RAG model flow.
- The FastAPI service is designed to work with GPUs but can run on CPU with lower performance.
- The RAG system uses `SentenceTransformer` and `ChromaDB` for similarity retrieval.

## License

See `LICENSE` for licensing details.
