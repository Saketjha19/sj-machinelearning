import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# Create pivot table for collaborative filtering
user_movie_ratings = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_movie_ratings)
distances, indices = knn.kneighbors(user_movie_ratings, n_neighbors=10)

# Map movieId to index for retrieval
movieId_to_index = {movie_id: idx for idx, movie_id in enumerate(user_movie_ratings.index)}

def get_collab_recommendations(movie_id):
    if movie_id not in movieId_to_index:
        return []  # Return empty if movie not found
    row_idx = movieId_to_index[movie_id]
    neighbors = indices[row_idx][1:]  # Skip the first as it's the movie itself
    return user_movie_ratings.iloc[neighbors].index.tolist()

# Streamlit UI
st.title("Movie Recommender System")
st.write("Enter a movie ID to get recommendations.")

movie_id = st.number_input("Movie ID", min_value=int(movies['movieId'].min()), max_value=int(movies['movieId'].max()), step=1)

if st.button("Get Recommendations"):
    recommendations = get_collab_recommendations(movie_id)
    if recommendations:
        recommended_movies = movies[movies['movieId'].isin(recommendations)][['movieId', 'title']]
        st.write("Recommended Movies:")
        st.dataframe(recommended_movies)
    else:
        st.write("No recommendations found.")
