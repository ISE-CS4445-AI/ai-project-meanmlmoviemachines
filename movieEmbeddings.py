import faiss
import numpy as np

# Load trained movie embeddings
movie_factors = np.load("movie_factors.npy")

# Create FAISS index
index = faiss.IndexFlatL2(movie_factors.shape[1])  # L2 distance for similarity
index.add(movie_factors)

# Save index
faiss.write_index(index, "faiss_movie_index.bin")
