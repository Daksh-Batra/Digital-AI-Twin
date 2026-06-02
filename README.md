# Digital-AI-Twin
##Richard Hamming Digital Twin

A Digital Twin of Richard Hamming built using RAG (Retrieval Augmented Generation), Gemini 2.5 Flash, ChromaDB and Streamlit. The chatbot answers questions in Richard Hamming's style using information from his books, lectures, papers and biography.

##Features
-RAG-based retrieval from Richard Hamming's works
-Persona-based responses
-Short-term conversation memory
-Long-term memory across sessions
-Streamlit interface

##How to Run
1. Install dependencies using pip install -r requirements.txt
2. Add your Gemini API key in .env
3. Run python rag/ingest.py
4. Run streamlit run app.py
