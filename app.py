import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Constants -----------------
API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMG = "https://via.placeholder.com/300x450?text=No+Image"

# --------------- Streamlit Page Config ---------------
st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎬 Hybrid Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Mix of <strong>Content Similarity</strong> and <strong>Popularity</strong> to give you the best picks!</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------- Function: Search TMDb ---------------
def search_movie(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

# --------------- Function: Get Genres ---------------
def get_genres():
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    genres = response.json().get("genres", [])
    return {g['id']: g['name'] for g in genres}

# --------------- Function: Autocomplete ---------------
def get_suggestions(prefix):
    results = search_movie(prefix)
    suggestions = [m["title"] for m in results][:10]
    return list(dict.fromkeys(suggestions))

# --------------- Function: Hybrid Recommender ---------------
def hybrid_recommend(title, alpha=0.5, genre_filter=None, year_range=None, min_rating=0):
    results = search_movie(title)
    if not results:
        return [], None

    target = results[0]
    all_movies = results[:20]  # increase scope a bit

    # Apply filters
    filtered = []
    for m in all_movies:
        release_year = int(m['release_date'][:4]) if m.get('release_date') else None
        if (
            (genre_filter and genre_filter not in m.get('genre_ids', [])) or
            (year_range and (release_year is None or not (year_range[0] <= release_year <= year_range[1]))) or
            (m.get('vote_average', 0) < min_rating)
        ):
            continue
        filtered.append(m)

    if len(filtered) < 2:
        return [], target['title']

    titles = [m['title'] for m in filtered]
    overviews = [m.get('overview', '') for m in filtered]
    votes = [m.get('vote_average', 0) for m in filtered]
    posters = [IMAGE_BASE_URL + m['poster_path'] if m.get('poster_path') else PLACEHOLDER_IMG for m in filtered]

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

# --------------- UI: Movie Input & Suggestions ---------------
st.markdown("### 🎯 Enter a movie name to get started:")
user_input = st.text_input("Type to search...", label_visibility="collapsed", placeholder="Start typing a movie like Inception, Titanic...")

suggestions = get_suggestions(user_input) if user_input else []
selected_movie = st.selectbox("Pick a movie from suggestions:", suggestions) if suggestions else None

# --------------- Filter Options ---------------
st.markdown("---")
st.markdown("### 🔎 Optional Filters to Customize Your Recommendations:")

genre_dict = get_genres()
genre_name_to_id = {v: k for k, v in genre_dict.items()}
genre_choice = st.selectbox("🎭 Filter by Genre (Optional)", ["None"] + list(genre_name_to_id.keys()))
genre_id = genre_name_to_id.get(genre_choice) if genre_choice != "None" else None

year_range = st.slider("📅 Filter by Release Year", 1950, 2025, (2000, 2025))
min_rating = st.slider("⭐ Minimum Rating", 0.0, 10.0, 5.0, step=0.5)
alpha = st.slider("⚖️ Balance: Content Similarity vs Popularity", 0.0, 1.0, 0.5)

# --------------- Display Recommendations ---------------
if selected_movie:
    recommended, match_title = hybrid_recommend(
        selected_movie,
        alpha=alpha,
        genre_filter=genre_id,
        year_range=year_range,
        min_rating=min_rating
    )
    if recommended:
        st.success(f"📌 Recommendations based on: **{match_title}**")
        st.markdown("### 🔥 Top Picks for You:")

        cols = st.columns(5)
        for i, (rec_title, rec_poster) in enumerate(recommended):
            with cols[i % 5]:
                st.image(rec_poster, use_container_width=True)
                st.markdown(f"<p style='text-align:center'><strong>{rec_title}</strong></p>", unsafe_allow_html=True)
    else:
        st.warning("😕 No results found for selected filters. Try relaxing them.")
