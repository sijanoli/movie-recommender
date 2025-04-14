import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"

st.title("🎬 Real-Time Movie Recommender (Hybrid)")

# --- Function: Search movie on TMDb ---
def search_movie(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

# --- Function: Get content + collaborative scores ---
def hybrid_recommend(title, alpha=0.5):
    results = search_movie(title)
    if not results:
        return [], None

    # Get top 10 similar movies based on overview
    target = results[0]
    all_movies = results[:10]

    titles = [m['title'] for m in all_movies]
    overviews = [m.get('overview', '') for m in all_movies]
    votes = [m.get('vote_average', 0) for m in all_movies]
    posters = [f"https://image.tmdb.org/t/p/w200{m.get('poster_path', '')}" for m in all_movies]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(overviews)
    content_sim = cosine_similarity(tfidf_matrix)

    idx = 0  # reference movie is the first one
    content_scores = content_sim[idx]

    # Normalize vote averages to 0–1
    vote_scores = pd.Series(votes).fillna(0)
    vote_scores = (vote_scores - vote_scores.min()) / (vote_scores.max() - vote_scores.min())

    # Combine both scores (hybrid)
    final_scores = alpha * content_scores + (1 - alpha) * vote_scores
    sorted_idx = final_scores.argsort()[::-1][1:6]

    recommendations = [(titles[i], posters[i]) for i in sorted_idx]
    return recommendations, target['title']

# --- Streamlit Interface ---
movie_title = st.text_input("Enter a movie name:")
alpha = st.slider("Content vs Popularity Balance (α)", 0.0, 1.0, 0.5)

if movie_title:
    recommended, match_title = hybrid_recommend(movie_title, alpha=alpha)
    if recommended:
        st.success(f"Showing results based on: **{match_title}**")
        for i, (rec_title, rec_poster) in enumerate(recommended, 1):
            st.image(rec_poster, width=100)  # Displaying the poster
            st.write(f"{i}. 🎥 {rec_title}")
    else:
        st.warning("No recommendations found.")
