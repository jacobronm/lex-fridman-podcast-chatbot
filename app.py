import streamlit as st
import pandas as pd
import kagglehub
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Lex Fridman Podcast Chatbot")

st.title("Lex Fridman Podcast Chatbot")

st.write(
    "This chatbot retrieves podcast transcript content using sentence embeddings "
    "and semantic similarity search."
)

# load dataset
@st.cache_data
def load_data():

    path = kagglehub.dataset_download(
        "asaniczka/lex-fridman-podcast-transcript"
    )

    df = pd.read_csv(path + "/transcript.csv")

    return df

df = load_data()

# extract text
texts = df["text"].dropna().tolist()

# use smaller sample for deployment speed
sample_texts = texts[:300]

# load embedding model
@st.cache_resource
def load_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model

model = load_model()

# create embeddings
@st.cache_resource
def create_embeddings(texts):

    embeddings = model.encode(texts)

    return embeddings

embeddings = create_embeddings(sample_texts)

# retrieval function
def retrieve(query, top_k=3):

    query_embedding = model.encode([query])[0]

    similarities = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = [sample_texts[i] for i in top_indices]

    return results

# chatbot UI
query = st.text_input("Ask a question about the podcast:")

if query:

    docs = retrieve(query)

    answer = " ".join(docs)

    st.subheader("Answer")

    st.write(answer[:2000])

    st.subheader("Retrieved Transcript Chunks")

    for i, doc in enumerate(docs):

        st.write(f"Chunk {i+1}")

        st.write(doc[:1000])
