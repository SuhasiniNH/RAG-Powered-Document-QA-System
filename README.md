# RAG-Powered-Document-QA-System
RAG connects LLMs with a company's knowledge base to enhance the accuracy of outputs.
Instead of fine-tuning the entire language model with the new corpus, RAG leverages the power of retrieval to access relevant information on demand. By combining retrieval mechanisms with language models, RAG enhances the responses by incorporating external context. This external context can be provided as a vector embedding.

Architecture
PyPDFLoader - Load PDF documents
RecursiveCharacterTextSplitter - Split into chunks
OpenAIEmbeddings - Convert text to vectors
ChromaDB - Store vectors locally
Retriever - Find similar chunks
ChatOpenAI - Generate answers
RAG Chain - Connect everything
Streamlit - Build UI
dotenv - Manage API keys
@st.cache_resource - Cache for performance
