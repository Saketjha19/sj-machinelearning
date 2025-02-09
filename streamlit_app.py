import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movie_merge = movies.merge(ratings, on="movieId")

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
def get_recommendations(movie_name, model, data, n=5):
    movie_index = data[data['title'] == movie_name].index[0]
    distances, indices = model.kneighbors(data.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=n+1)
    recommendations = [data.iloc[i].title for i in indices.flatten()[1:]]
    return recommendations

if st.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(selected_movie, knn_model, movies)
        st.write("Recommended Movies:")
        for rec in recommendations:
            st.write(f"- {rec}")
    except Exception as e:
        st.error(f"Error: {e}")
