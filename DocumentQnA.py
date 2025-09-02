import asyncio
import nest_asyncio

import streamlit as st
import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ---- Fix for Streamlit + async gRPC ----
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()
# ----------------------------------------

load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))


st.set_page_config(page_title='RAG with Langchain',page_icon='🤖',layout="wide")
st.title("📂 RAG app QnA with Langchain")
uploaded_file = st.file_uploader("Upload a PDF or TXT file or DOCX file",type=['.pdf','.txt','docx'])


if uploaded_file:
    with st.spinner('File uploading.....Please wait⏳'):
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Now, safely check the file extension
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstores = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstores.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        st.success("File Processed Successfully!!")


        user_query=st.text_input("User Query")

        if user_query:
            if st.button("Get Answer"):
                with st.spinner("Fetching Answer...Please Wait"):
                    response = qa(user_query)
                    st.subheader("Answer")
                    st.write(response)
             