import streamlit as st
import pandas as pd
import kagglehub
import chromadb

from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Lex Fridman Podcast Chatbot")

st.title("Lex Fridman Podcast Chatbot")

st.write(
    "This app follows the same workflow used in the notebook: dataset loading, "
    "text splitting, embeddings, ChromaDB vector storage, retrieval, and basic "
    "RAG-style answer generation."
)

@st.cache_resource
def create_vector_database():
    dataset_path = kagglehub.dataset_download(
        "rajneesh231/lex-fridman-podcast-transcript"
    )

    df = pd.read_csv(dataset_path + "/podcastdata_dataset.csv")
    texts = df["text"].dropna().tolist()

    chunks = []

    chunk_size = 1000
    overlap = 200

    for text in texts:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

    sample_chunks = chunks[:500]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sample_chunks)

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="podcast")

    if collection.count() == 0:
        for i, emb in enumerate(embeddings):
            collection.add(
                embeddings=[emb.tolist()],
                documents=[sample_chunks[i]],
                ids=[str(i)]
            )

    return model, collection

def retrieve(query, top_k=3):
    model, collection = create_vector_database()

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results["documents"][0]

def generate_answer(query):
    docs = retrieve(query)

    context = " ".join(docs)

    answer = f"Based on the podcast transcripts: {context[:1000]}"

    return answer

query = st.text_input("Ask a question about the Lex Fridman Podcast:")

if query:
    with st.spinner("Searching podcast transcripts..."):
        answer = generate_answer(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("User Question")
    st.write(query)
