import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

st.set_page_config(page_title="Information Retrieval", page_icon="🔎")

st.title("🔎 Information Retrieval System")
st.subheader("Search Reuters News Articles Using Word Embeddings")

@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = f.readlines()
    return embeddings, documents

@st.cache_resource
def load_model():
    return Word2Vec.load("w2v.model")

embeddings, documents = load_data()
w2v = load_model()

st.write("Loading Reuters dataset...")
st.write(f"Loaded {len(documents)} documents from Reuters.")
st.write("Training Word2Vec model...")
st.write(f"Vocabulary size: {len(w2v.wv.index_to_key)}")

def embed_text(text: str, model: Word2Vec):
    tokens = simple_preprocess(text)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def retrieve_top_k(query_vec, embeddings, k=10):
    sims = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_idx]

query = st.text_input("Enter your search query:")

if st.button("Search") and query.strip():
    qvec = embed_text(query, w2v)
    results = retrieve_top_k(qvec, embeddings, k=10)

    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- (Score: {score:.4f}) {doc.strip()[:300]}...")
else:
    st.info("Type a query and click Search.")