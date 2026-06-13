from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory = "chroma_db", embedding_function = embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

query = "Who is Richard Hamming?"

results = db.similarity_search(query,k=3)

context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""You are Richard Hamming.
Answer using the context given below.

Context: {context}

Question: {query}

"""

response = llm.invoke(prompt)

print(response.content)