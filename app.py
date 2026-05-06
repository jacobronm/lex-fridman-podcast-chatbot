import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import zipfile

st.set_page_config(page_title="Lex Fridman Podcast Chatbot")

st.title("Lex Fridman Podcast Chatbot")

st.write(
    "This deployed app uses the Lex Fridman transcript dataset, text chunks, "
    "sentence embeddings, and semantic retrieval. The notebook contains the full "
    "ChromaDB-based RAG workflow."
)

@st.cache_data
def load_data():
    with zipfile.ZipFile("archive.zip", "r") as zip_ref:
        zip_ref.extractall("data")

    df = pd.read_csv("data/podcastdata_dataset.csv")
    return df

@st.cache_resource
def build_embeddings():
    df = load_data()
    texts = df["text"].dropna().tolist()

    chunks = []
    chunk_size = 1000
    overlap = 200

    for text in texts:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

    sample_chunks = chunks[:500]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sample_chunks)

    return model, embeddings, sample_chunks

model, embeddings, sample_chunks = build_embeddings()

def retrieve(query, top_k=3):
    query_embedding = model.encode([query])[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [sample_chunks[i] for i in top_indices]

def generate_answer(query):
    docs = retrieve(query)
    context = " ".join(docs)
    return f"Based on the podcast transcripts: {context[:1000]}"

query = st.text_input("Ask a question about the Lex Fridman Podcast:")

if query:
    with st.spinner("Searching podcast transcripts..."):
        answer = generate_answer(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Transcript Chunks")
    for i, doc in enumerate(retrieve(query), start=1):
        st.write(f"Chunk {i}")
        st.write(doc[:700])

    st.subheader("User Question")
    st.write(query)
