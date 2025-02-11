from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import faiss
import pandas as pd

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load precomputed model data
movie_factors = np.load("movie_factors.npy")
index = faiss.read_index("faiss_movie_index.bin")

# Mapping from MovieLens movie IDs to TMDb IDs
links_df = pd.read_csv("links.csv")
lens_to_tmb = dict(zip(links_df["movieId"], links_df["tmdbId"]))
tmb_to_lens = dict(zip(links_df["tmdbId"], links_df["movieId"]))
# Movies ordering
movies_df = pd.read_csv("movies.csv")
movie_ids = movies_df["movieId"].tolist()

# Pydantic model for the request body
class RecommendationRequest(BaseModel):
    watched_movie_indices: list[int]
    top_k: int = 10

# Handle OPTIONS request for preflight checks
@app.options("/recommend")
async def options_recommend():
    return {"message": "CORS preflight response"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# POST request to fetch movie recommendations

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    watched_movie_indices = request.watched_movie_indices
    top_k = request.top_k

    # Filter out movies that are not in the database or out of bounds
    valid_indices = [idx for idx in watched_movie_indices if idx in tmb_to_lens and tmb_to_lens[idx] < len(movie_factors)]

    if not valid_indices:
        return {"message": "No recommendations found"}

    # Aggregate the vectors of watched movies
    watched_movie_vecs = np.array([movie_factors[tmb_to_lens[idx]] for idx in valid_indices])
    user_profile = np.mean(watched_movie_vecs, axis=0).reshape(1, -1)

    distances, indices = index.search(user_profile, top_k)
    recommended_indices = indices.tolist()[0]

    recommended_tmdb_ids = []
    for idx in recommended_indices:
        ml_id = movie_ids[idx]
        tmdb_id = lens_to_tmb.get(ml_id)
        if tmdb_id is not None:
            recommended_tmdb_ids.append(tmdb_id)
        else:
            recommended_tmdb_ids.append(ml_id)

    if not recommended_tmdb_ids:
        return {"message": "No recommendations found"}

    return {"recommended_movie_ids": recommended_tmdb_ids}