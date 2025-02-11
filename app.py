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
    movie_index: int
    top_k: int = 10

# Handle OPTIONS request for preflight checks
@app.options("/recommend")
async def options_recommend():
    return {"message": "CORS preflight response"}

# POST request to fetch movie recommendations
@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    movie_index = request.movie_index
    movie_lens_id = tmb_to_lens.get(movie_index)
    top_k = request.top_k

    movie_vec = movie_factors[movie_lens_id].reshape(1, -1)
    distances, indices = index.search(movie_vec, top_k)
    recommended_indices = indices.tolist()[0]

    recommended_tmdb_ids = []
    for idx in recommended_indices:
        ml_id = movie_ids[idx]
        tmdb_id = lens_to_tmb.get(ml_id)
        if tmdb_id is not None:
            recommended_tmdb_ids.append(tmdb_id)
        else:
            recommended_tmdb_ids.append(ml_id)

    return {"recommended_movie_ids": recommended_tmdb_ids}
