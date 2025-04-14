import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Constants --------------------
API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMG = "https://via.placeholder.com/300x450?text=No+Image"

# -------------------- Page Config --------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .desc {
            text-align: center;
            color: gray;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎬 Hybrid Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Smart suggestions using content similarity & popularity</div>", unsafe_allow_html=True)

# -------------------- TMDb API Functions --------------------
def search_movie(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

def get_suggestions(prefix):
    if not prefix:
        return []
    results = search_movie(prefix)
    return list({m["title"] for m in results})[:10]

# -------------------- Recommendation Logic --------------------
def hybrid_recommend(title, alpha=0.5, year_range=(1950, 2025), min_rating=0):
    results = search_movie(title)
    if not results:
        return [], None

    target = results[0]
    all_movies = results[:20]

    filtered = []
    for m in all_movies:
        release_year = int(m['release_date'][:4]) if m.get('release_date') else None
        if release_year and not (year_range[0] <= release_year <= year_range[1]):
            continue
        if m.get('vote_average', 0) < min_rating:
            continue
        filtered.append(m)

    if len(filtered) < 2:
        return [], target['title']

    titles = [m['title'] for m in filtered]
    overviews = [m.get('overview', '') for m in filtered]
    votes = [m.get('vote_average', 0) for m in filtered]

    posters = []
    for m in filtered:
        path = m.get('poster_path')
        if path:
            posters.append(f"{IMAGE_BASE_URL}{path}")
        else:
            posters.append(PLACEHOLDER_IMG)

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

# -------------------- Layout --------------------
left, right = st.columns([3, 1])  # 75% left, 25% right

# -------- Left Side (Recommendations) --------
with left:
    st.markdown("### 🎯 Search for a movie")
    movie_input = st.text_input("Type movie name:", placeholder="e.g., Titanic, Inception...")

    suggestions = get_suggestions(movie_input)
    selected_movie = st.selectbox("Suggestions:", suggestions) if suggestions else None

    if selected_movie:
        recommendations, match_title = hybrid_recommend(
            selected_movie,
            alpha=st.session_state.get("alpha", 0.5),
            year_range=st.session_state.get("year_range", (1950, 2025)),
            min_rating=st.session_state.get("min_rating", 0)
        )

        if recommendations:
            st.success(f"📌 Recommendations based on: **{match_title}**")
            cols = st.columns(5)
            for i, (rec_title, rec_poster) in enumerate(recommendations):
                with cols[i % 5]:
                    st.image(rec_poster, use_container_width=True)
                    st.markdown(f"<div style='text-align:center; font-weight:600'>{rec_title}</div>", unsafe_allow_html=True)
        else:
            st.warning("😕 No matching recommendations found.")

# -------- Right Side (Filters) --------
with right:
    st.markdown("### 🎛️ Filters")
    year_range = st.slider("📅 Year", 1950, 2025, (2000, 2025))
    min_rating = st.slider("⭐ Min Rating", 0.0, 10.0, 5.0, 0.5)
    alpha = st.slider("⚖️ Content vs Popularity", 0.0, 1.0, 0.5)

    # Store in session for use on left
    st.session_state["year_range"] = year_range
    st.session_state["min_rating"] = min_rating
    st.session_state["alpha"] = alpha
