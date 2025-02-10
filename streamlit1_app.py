import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movie_dataset.csv")

# Fill missing values
df.fillna("", inplace=True)

# Define features to use for recommendations
features = ['genres', 'keywords', 'title', 'cast', 'director']

def combine_features(row):
    return " ".join(row[feature] for feature in features)

df["combined_features"] = df.apply(combine_features, axis=1)

# Convert text data into numerical form
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(df["combined_features"])

# Compute similarity between movies
cosine_sim = cosine_similarity(feature_matrix)

def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in df['title'].values:
        return []
    
    movie_idx = df[df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    recommended_movies = [df.iloc[i[0]]['title'] for i in sorted_scores]
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.write("Enter a movie title to get recommendations!")

selected_movie = st.selectbox("Select a movie:", df['title'].unique())

if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_movie)
    if recommendations:
        st.write("### Recommended Movies:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("No recommendations found.")
