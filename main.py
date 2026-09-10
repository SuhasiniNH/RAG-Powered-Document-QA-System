from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import FileResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai_key = os.getenv("ApiKey")

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====== RAG Setup ======
def load_rag_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=openai_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                     openai_api_key=openai_key)

    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        vector_db = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)
    else:
        loader = PyPDFLoader("Pandas Notes.pdf")
        doc = loader.load()

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(doc)

        vector_db = Chroma.from_documents(chunks, embeddings,
                                         persist_directory=persist_directory)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using the context. Say 'I don't know' if unsure.\n\n{context}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

rag_chain = load_rag_chain()


# Serve index.html at root
@app.get("/")
async def root():
    return FileResponse("static/index.html")

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(payload: Question):
    response = rag_chain.invoke({"input": payload.question})
    return {
        "answer": response["answer"],
        "context": response.get("context", "")
    }

