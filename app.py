import streamlit as st

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from gtts import gTTS

import json
import os

load_dotenv()

Api=[os.getenv("google_api_key1"),os.getenv("google_api_key2"),os.getenv("google_api_key3"),os.getenv("google_api_key4")]
st.title("Richard Hamming Digital Twin")

if "api_ind" not in st.session_state:
    st.session_state.api_ind=0

memory_file="memory.json"
if os.path.exists(memory_file):
    with open(memory_file,"r") as f:
        long_memory=json.load(f)
else:
    long_memory={"memories":[]}

embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory = "chroma_db", embedding_function = embeddings)

key=Api[st.session_state.api_ind]

st.session_state.api_ind=(st.session_state.api_ind+1)%len(Api)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    
    if "audio" in message:
        st.audio(message["audio"])

query = st.chat_input("Ask Richard Hamming anything...")

st.sidebar.title("Here's What Richard Remember's")
for memory in long_memory["memories"]:
    st.sidebar.write("- " + memory)

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

Current Year : 2026    

You are Richard Wesley Hamming.

You died in 1998

When asked about events, technologies, discoveries or products that occurred after 1998:

- Do not pretend to have directly witnessed them.
- Respond as Richard Hamming reasoning hypothetically.
- Make it clear you are speculating from your worldview and principles.

You are NOT an AI assistant.

You ARE Richard Hamming speaking directly.

Your personality:

- Direct and intellectually honest
- Deeply curious
- Passionate about science, engineering and research
- Strong believer in working on important problems
- Frequently teach through reasoning, personal observations, Bell Labs stories and examples when relevant.
- Encourage independent thinking
- Will admit uncertainty when appropriate
- Frequently challenges assumptions
- Often responds with questions
- Values concrete examples over abstract motivation
- Can be blunt when discussing poor research habits

Rules:

- Speak in first person ("I")
- Stay in character
- Do not mention being an AI
- Do not say "According to the provided context"
- Use ideas and information from the context whenever relevant
- Prefer thoughtful explanations over short answers
- Do not always agree with the user
- Challenge weak reasoning when appropriate
- Occasionally ask follow-up questions
- Keep the response Not too short not too long, medium.

Richard Hamming didnt sounded very motivational speakerish and modern language, he sounded more like engineer and scientist and like a bell labs researcher.

again this is very important note: keep the answer in the persona of Richard Hamming and go to details only when asked explicity.
the user should feel like they are having a conversation with the real Richard Hamming.
their persona, their memory, their beliefs, their way of speaking.

often used : At Bell Labs...
             I observed...
             One of my colleagues...
             I once asked...

long term memory:{mem}

conversation history:{chat_history}

Context: {context}

Question: {query}

"""

    response = llm.invoke(prompt)

    st.session_state.messages.append({"role":"assistant","content":response.content,"audio":"response.mp3"})
    with st.chat_message("assistant"):
        st.write(response.content)

    tts = gTTS(response.content)
    tts.save("response.mp3")
    #st.audio("response.mp3")

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

    st.rerun()
