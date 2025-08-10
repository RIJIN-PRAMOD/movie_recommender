import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process  


API_KEY = st.secrets["TMDB_API_KEY"]


def fetch_poster_and_link(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    poster_url = "https://image.tmdb.org/t/p/w500" + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image"
    tmdb_link = f"https://www.themoviedb.org/movie/{movie_id}"
    return poster_url, tmdb_link


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    recommended_links = []
    
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        poster_url, tmdb_link = fetch_poster_and_link(movie_id)
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(poster_url)
        recommended_links.append(tmdb_link)
    
    return recommended_movies, recommended_posters, recommended_links


movies_df = pd.read_csv("https://drive.google.com/uc?export=download&id=1yMHBZcD52z45VyApdogBSBvIrow9CTAu
")
credits_df = pd.read_csv("https://drive.google.com/uc?export=download&id=1XgzxRLnpF5zlaeOfyGpdD_CmcKXtJThQ")
movies = movies_df.merge(credits_df, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.fillna('', inplace=True)

def clean_data(x):
    return x.replace(" ", "").replace("-", "").lower()

movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['crew']
movies['tags'] = movies['tags'].apply(clean_data)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)


st.title("🎬 Movie Recommendation ")

user_input = st.text_input("Type a movie name (partial allowed):", "")

if user_input:
    matches = process.extract(user_input, movies['title'], limit=5, score_cutoff=60)
    if matches:
        matched_titles = [m[0] for m in matches]
        selected_movie = st.selectbox("Did you mean:", matched_titles)
        
        if st.button("Recommend"):
            names, posters, links = recommend(selected_movie)
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.markdown(f"**{names[i]}**")
                    st.markdown(f"[![poster]({posters[i]})]({links[i]})", unsafe_allow_html=True)
    else:
        st.warning("No close matches found. Try another title.")


