<div align="center">

# 🩺 Medical Chatbot (RAG-based)

Context-aware Q&A over a local medical knowledge base using FAISS vector search + HuggingFace or Groq-hosted LLMs.

</div>

## 🧠 Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that lets you ask medical questions grounded in a curated PDF knowledge base (e.g. an encyclopedia of medicine). Instead of hallucinating, the LLM is constrained by retrieved passages from a FAISS vector store built from your documents.

Two primary entry points:
- `connect_mem_llm.py` – CLI prototype using a HuggingFace Inference endpoint (e.g. Mistral 7B Instruct).
- `medi.py` – Streamlit chat UI using a Groq-hosted model (Llama 4 Maverick) with retrieval.

## ✨ Key Features
- FAISS vector store for fast semantic retrieval
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`) – switchable to remote API mode
- Modular prompt template injection
- Groq or HuggingFace LLM backends
- Source document traceability (shows which chunks supported the answer)
- Caching of vector store + embeddings via Streamlit resource cache

## 🏗 Architecture
```
PDF(s) --> Text Splitter --> Embeddings --> FAISS Index (vectorstore/db_faiss)
								│
User Query --> Retriever (top-k) ---------------┘
			    │
		    Prompt Assembly
			    │
		    LLM Generation (HF or Groq)
			    │
		    Answer + Source Chunks
```

### Main Components
| File | Role |
|------|------|
| `create_mem_llm.py` | Builds FAISS index from PDFs (embedding + persist) | ->(basically creates memory for llm)
| `connect_mem_llm.py` | CLI RAG query using HuggingFaceEndpoint | ->(connects memory with llm)
| `medi.py` | Streamlit chat interface using Groq Chat model + FAISS retrieval |
| `vectorstore/db_faiss` | Persisted FAISS index (created beforehand) |
| `data/` | PDF source documents |

## 🔐 Environment Variables
Create a `.env` file :
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=groq_xxxxxxxxxxxxxxxxxxx
```

Optional (future expansion):
```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_REPO_ID=mistralai/Mistral-7B-Instruct-v0.3
```

## ⚙️ Installation
Use the provided `requirements.txt` 

### 1. Create & Activate Virtual Environment
```zsh
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```zsh
pip install --upgrade pip
pip install -r requirements.txt
```

If using `uv`:
```zsh
uv sync
```

### 3. Set Environment Variables
```zsh
export HF_TOKEN=hf_...yourtoken...
export GROQ_API_KEY=groq_...yourtoken...
```
Or create a `.env` file and rely on `dotenv` where enabled.

## 🗂 Building the Vector Store
If you have not yet created `vectorstore/db_faiss`, run the memory creation script :
```zsh
python create_mem_llm.py or uv run create_mem_llm.py
```
This should:
1. Load PDFs from `data/`
2. Chunk text
3. Embed chunks using `HuggingFaceEmbeddings`
4. Persist FAISS index under `vectorstore/db_faiss`

If the file does not yet exist, implement a pipeline similar to:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, emb)
db.save_local("vectorstore/db_faiss")
```

## 💬 Running the CLI Version
```zsh
source .venv/bin/activate
export HF_TOKEN=...  # if not in .env
python connect_mem_llm.py
```
Enter a query at the prompt: `How is hypertension managed?`

## 🖥 Running the Streamlit App
```zsh
source .venv/bin/activate
export GROQ_API_KEY=groq_...  # if not in .env
streamlit run medi.py or uv run streamlit run medi.py
```
Open the URL shown (default: http://localhost:8501) and start chatting.


## 🧱 Extending
- Add multi-PDF ingestion (glob over `data/*.pdf`)
- Enable streaming tokens in UI
- Add OpenAI / Anthropic backend abstraction
- Persist chat history with sources
- Add evaluation harness (e.g. RAGAS) for answer faithfulness

## ⚖️ Disclaimer
This tool is for educational and reference purposes only. It does **not** provide medical advice, diagnosis, or treatment recommendations. Always consult a licensed healthcare professional for medical decisions.




