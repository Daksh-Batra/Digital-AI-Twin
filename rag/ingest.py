from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

import os

load_dotenv()

data_folder = "data"
Chroma_Path="chroma_db"
documents =[]

for root,dirs,files in os.walk(data_folder):
    for file in files:
        if file.endswith(".pdf"):
            path = os.path.join(root, file)
            print(f"loading {file}...")
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

print(f"loaded {len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=Chroma_Path)

print("vector database created")