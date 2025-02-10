import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge datasets
movie_merge = movies.merge(ratings, on="movieId").drop(["genres", "timestamp"], axis=1)

# Create pivot table
movie_users = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)

# Convert to sparse matrix
mat_movies = csr_matrix(movie_users.values)

# Train KNN model
model = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1)
model.fit(mat_movies)

def get_movie_recommendations(movie_title):
    if movie_title not in movies['title'].values:
        return []
    
    movie_id = movies[movies['title'] == movie_title]['movieId'].iloc[0]
    movie_idx = list(movie_users.index).index(movie_id)
    distances, indices = model.kneighbors([movie_users.iloc[movie_idx]])
    
    recommended_movies = [movies[movies['movieId'] == movie_users.index[i]]['title'].values[0] for i in indices[0][1:]]
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommender System (KNN-Based)")
st.write("Enter a movie title to get recommendations!")

selected_movie = st.selectbox("Select a movie:", movies['title'].unique())

if st.button("Get Recommendations"):
    recommendations = get_movie_recommendations(selected_movie)
    if recommendations:
        st.write("### Recommended Movies:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("No recommendations found.")
