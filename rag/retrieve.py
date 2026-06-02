from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db=Chroma(persist_directory='chroma_db',embedding_function=embeddings)

query = "Who is Richard Hamming?"

results = db.similarity_search(query,k=3) #top 3 similar chunk results

for i ,doc in enumerate(results):
    print(f"Result {i+1}")
    print(doc.page_content[:1000])