import streamlit as st

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

import json
import os

load_dotenv()
st.title("Richard Hamming Digital Twin")

memory_file="memory.json"
if os.path.exists(memory_file):
    with open(memory_file,"r") as f:
        long_memory=json.load(f)
else:
    long_memory={"memories":[]}

embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory = "chroma_db", embedding_function = embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask Richard Hamming anything...")



if query:
    st.session_state.messages.append({"role":"user","content":query})

    with st.chat_message("user"):
        st.write(query)
    
    results = db.similarity_search(query,k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    chat_history=""
    for message in st.session_state.messages:
        chat_history+=f"{message["role"]}: {message["content"]}\n"
    
    mem="\n".join(long_memory["memories"])

    prompt = f"""
    
You are Richard Wesley Hamming.

You are NOT an AI assistant.

You ARE Richard Hamming speaking directly.

Your personality:

- Direct and intellectually honest
- Deeply curious
- Passionate about science, engineering and research
- Strong believer in working on important problems
- Frequently teach through reasoning and examples
- Encourage independent thinking
- Will admit uncertainty when appropriate

Rules:

- Speak in first person ("I")
- Stay in character
- Do not mention being an AI
- Do not say "According to the provided context"
- Use ideas and information from the context whenever relevant
- Prefer thoughtful explanations over short answers
-Keep answers concise unless the user explicitly asks for detail.
-Default to 2-4 paragraphs.

long term memory:{mem}

conversation history:{chat_history}

Context: {context}

Question: {query}

"""

    response = llm.invoke(prompt)

    st.session_state.messages.append({"role":"assistant","content":response.content})
    with st.chat_message("assistant"):
        st.write(response.content)

    memory_promt=f"""
Extract long-term facts about the user.

Only save stable information such as:
- interests
- goals
- background
- preferences
- career aspirations

Do NOT save:
- temporary questions
- greetings
- one-time requests

If there is nothing worth remembering, reply ONLY:

NONE

User Message:
{query}
"""
    mem_response = llm.invoke(memory_promt)
    n_mem=mem_response.content
    if(n_mem!="NONE" and n_mem not in long_memory["memories"]):
        long_memory["memories"].append(n_mem)
    
        with open(memory_file,"w") as f:
            json.dump(long_memory,f)
