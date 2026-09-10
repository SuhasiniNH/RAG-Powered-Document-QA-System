# RAG-Powered-Document-QA-System
RAG connects LLMs with a company's knowledge base to enhance the accuracy of outputs.
Instead of fine-tuning the entire language model with the new corpus, RAG leverages the power of retrieval to access relevant information on demand. By combining retrieval mechanisms with language models, RAG enhances the responses by incorporating external context. This external context can be provided as a vector embedding.

**Architecture**
PyPDFLoader - Load PDF documents
RecursiveCharacterTextSplitter - Split into chunks
OpenAIEmbeddings - Convert text to vectors
ChromaDB - Store vectors locally
Retriever - Find similar chunks
ChatOpenAI - Generate answers
RAG Chain - Connect everything
fastapi - Build backend
dotenv - Manage API keys

# Project Structure:
```text
RAG-Powered-Document-QA-System/
│── main.py                 # FastAPI backend (RAG chain + API endpoints)
│── requirements.txt        # Python dependencies
│── chroma_db/              # Local vector database (auto-created)
│── .env                    # Environment variables (OpenAI API key)
│
├── static/                 # Frontend assets
│   └── index.html          # UI with inline CSS + JS
│
└── data/                   # Documents to load
    └── Pandas Notes.pdf    # Example PDF for QA
