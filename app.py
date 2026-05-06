
import streamlit as st

st.set_page_config(page_title="Lex Fridman Podcast Chatbot")

st.title("Lex Fridman Podcast Chatbot")

st.write(
    "This is a simple Streamlit interface for the Lex Fridman Podcast chatbot. "
    "The notebook contains the full transcript loading, embedding, ChromaDB vector database, "
    "retrieval, and RAG-style answer generation process."
)

query = st.text_input("Ask a question about the Lex Fridman Podcast:")

if query:
    st.subheader("Answer")
    st.write(
        "This deployed version demonstrates the chatbot interface. "
        "In the notebook, the answer is generated using retrieved transcript chunks from the vector database."
    )

    st.subheader("User Question")
    st.write(query)
