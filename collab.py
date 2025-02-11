import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# Load dataset
df = pd.read_csv("ratings1.csv")

# Convert ratings to a sparse matrix
user_ids = {id: i for i, id in enumerate(df['userId'].unique())}
movie_ids = {id: i for i, id in enumerate(df['movieId'].unique())}
rows = df['userId'].map(user_ids)
cols = df['movieId'].map(movie_ids)
ratings = coo_matrix((df['rating'], (rows, cols)))

# Train ALS Model
model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=15)
model.fit(ratings)

# Save model
np.save("user_factors.npy", model.user_factors)
np.save("movie_factors.npy", model.item_factors)
