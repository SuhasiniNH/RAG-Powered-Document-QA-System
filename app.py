import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("ApiKey")

st.set_page_config(page_title="PDF QnA Bot", layout="wide")
st.title("PDF QnA Answer")

@st.cache_resource
def load_rag_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=openai_key)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                     openai_api_key=openai_key)

    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        print("Loading existing embeddings...")
        vector_db = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)
    else:
        print("Creating new embeddings...")
        loader =  PyPDFLoader("Pandas Notes.pdf")
        doc = loader.load()

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(doc)

        vector_db = Chroma.from_documents(chunks, embeddings,
                                         persist_directory=persist_directory)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt =  ChatPromptTemplate.from_messages([
        ("system", "Answer using the context. Say 'I dont know' if unsure. \n\n{context}"),
        ("human", "{input}"),
    ]) 

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

rag_chain = load_rag_chain()

# ========== UI ==========
st.sidebar.header("Settings")
max_tokens = st.sidebar.slider("Response Length", 100, 1000, 500)

# Input from user
user_question = st.text_area("Ask a question about the PDF:", 
                             placeholder="Get all the DataFrame Methods from the given PDF...")

if st.button("🔍 Get Answer"):
    if user_question:
        with st.spinner("Searching..."):
            response = rag_chain.invoke({"input": user_question})
            
            st.write(response["answer"])
            
            # Show sources
            if "context" in response:
                with st.expander("📚 Sources"):
                    st.write(response["context"])
    else:
        st.warning("Please enter a question!")

            