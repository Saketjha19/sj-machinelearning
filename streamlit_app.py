import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Load pre-trained model if available
try:
    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please train and save the model.")
    st.stop()

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Enter a movie name to get recommendations based on collaborative filtering.")

# User input for movie search
movie_list = movies['title'].tolist()
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Function to get recommendations
def get_recommendations(movie_name, model, data, ratings, n=5):
    # Get the movieId for the selected movie
    movie_id = data[data['title'] == movie_name]['movieId'].values[0]
    
    # Create the user-item matrix (pivot table)
    movie_features = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

    # Check if the movieId exists in the matrix
    if movie_id not in movie_features.index:
        st.error("Movie not found in rating data.")
        return []
    
    # Get index of movie in the feature matrix
    movie_index = movie_features.index.get_loc(movie_id)

    # Find nearest neighbors
    distances, indices = model.kneighbors(movie_features.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=n+1)

    # Fetch movie recommendations
    recommended_movie_ids = movie_features.iloc[indices.flatten()[1:]].index
    recommendations = data[data['movieId'].isin(recommended_movie_ids)]['title'].tolist()

    return recommendations

if st.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(selected_movie, knn_model, movies, ratings)
        st.write("Recommended Movies:")
        for rec in recommendations:
            st.write(f"- {rec}")
    except Exception as e:
        st.error(f"Error: {e}")

