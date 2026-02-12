import numpy as np
import nltk
from nltk.corpus import reuters
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Download Reuters (only needs to happen once)
nltk.download("reuters")
nltk.download("punkt")

print("Loading Reuters dataset...")

fileids = reuters.fileids()
documents = [reuters.raw(fid) for fid in fileids]

print(f"Loaded {len(documents)} documents from Reuters.")

# Tokenize
tokenized_docs = [simple_preprocess(doc) for doc in documents]

print("Training Word2Vec model...")

# Train Word2Vec
w2v = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,   # you can keep 100 (common for Word2Vec)
    window=5,
    min_count=2,
    workers=4
)

print("Vocabulary size:", len(w2v.wv.index_to_key))

# Build document embeddings by averaging word vectors
def doc_embedding(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

embeddings = np.vstack([doc_embedding(tokens, w2v) for tokens in tokenized_docs])

# Save everything
with open("documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.replace("\n", " ") + "\n")

np.save("embeddings.npy", embeddings)
w2v.save("w2v.model")

print("Saved documents.txt, embeddings.npy, w2v.model")
print("Embeddings shape:", embeddings.shape)