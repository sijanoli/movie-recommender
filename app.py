import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w200"

st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="wide")
st.title("🎬 Real-Time Movie Recommender (Hybrid)")

# --- Function: Search movie on TMDb ---
def search_movie(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

# --- Autocomplete movie titles ---
def get_suggestions(prefix):
    results = search_movie(prefix)
    suggestions = [m["title"] for m in results][:10]
    return list(dict.fromkeys(suggestions))  # remove duplicates

# --- Function: Get content + collaborative scores ---
def hybrid_recommend(title, alpha=0.5):
    results = search_movie(title)
    if not results:
        return [], None

    target = results[0]
    all_movies = results[:10]

    titles = [m['title'] for m in all_movies]
    overviews = [m.get('overview', '') for m in all_movies]
    votes = [m.get('vote_average', 0) for m in all_movies]
    posters = [IMAGE_BASE_URL + m.get('poster_path', '') if m.get('poster_path') else "" for m in all_movies]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(overviews)
    content_sim = cosine_similarity(tfidf_matrix)

    idx = 0
    content_scores = content_sim[idx]

    vote_scores = pd.Series(votes).fillna(0)
    vote_scores = (vote_scores - vote_scores.min()) / (vote_scores.max() - vote_scores.min())

    final_scores = alpha * content_scores + (1 - alpha) * vote_scores
    sorted_idx = final_scores.argsort()[::-1][1:6]

    recommendations = [(titles[i], posters[i]) for i in sorted_idx]
    return recommendations, target['title']

# --- UI: Live Suggestions ---
user_input = st.text_input("Start typing a movie name:")

suggestions = get_suggestions(user_input) if user_input else []
selected_movie = st.selectbox("Pick a movie from suggestions:", suggestions) if suggestions else None

alpha = st.slider("🎯 Content vs Popularity Balance (α)", 0.0, 1.0, 0.5)

# --- Show Recommendations ---
if selected_movie:
    recommended, match_title = hybrid_recommend(selected_movie, alpha=alpha)
    if recommended:
        st.success(f"🎯 Showing results based on: **{match_title}**")
        cols = st.columns(5)
        for i, (rec_title, rec_poster) in enumerate(recommended):
            with cols[i % 5]:
                if rec_poster:
                    st.image(rec_poster, use_column_width=True)
                st.caption(f"🎬 {rec_title}")
    else:
        st.warning("No recommendations found.")
