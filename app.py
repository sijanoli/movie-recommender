import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# API Configuration
API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"
POSTER_URL = "https://image.tmdb.org/t/p/w500"

# Streamlit UI
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender")
st.markdown("""
<style>
.recommendation-card {
    padding: 15px;
    border-radius: 10px;
    transition: transform 0.2s;
}
.recommendation-card:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# --- Core Functions ---
def get_movie_data(movie_id):
    """Fetch detailed movie data including similar movies (proxy for collaborative filtering)"""
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY, "append_to_response": "similar"}
    response = requests.get(url, params=params)
    return response.json()

def hybrid_recommend(movie_title, alpha=0.5):
    """Hybrid recommendation combining content and collaborative filtering"""
    # Step 1: Find the target movie
    search_url = f"{BASE_URL}/search/movie"
    search_params = {"api_key": API_KEY, "query": movie_title}
    search_results = requests.get(search_url, params=search_params).json().get("results", [])
    
    if not search_results:
        return [], None, None
    
    target_movie = search_results[0]
    target_id = target_movie["id"]
    
    # Step 2: Content-based filtering (using overview)
    movies = search_results[:10]  # Get initial set of similar movies
    overviews = [m.get("overview", "") for m in movies]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(overviews)
    content_sim = cosine_similarity(tfidf_matrix)[0]
    
    # Step 3: Collaborative filtering (using TMDB's similar movies as proxy)
    movie_data = get_movie_data(target_id)
    similar_movies = movie_data.get("similar", {}).get("results", [])[:10]
    
    # Combine both approaches
    all_movies = {}
    
    # Add content-based movies
    for idx, movie in enumerate(movies):
        all_movies[movie["id"]] = {
            "title": movie["title"],
            "poster": movie.get("poster_path"),
            "content_score": content_sim[idx],
            "collab_score": 0  # Initialize
        }
    
    # Add collaborative movies
    for movie in similar_movies:
        if movie["id"] in all_movies:
            all_movies[movie["id"]]["collab_score"] = movie.get("vote_average", 0)/10
        else:
            all_movies[movie["id"]] = {
                "title": movie["title"],
                "poster": movie.get("poster_path"),
                "content_score": 0,
                "collab_score": movie.get("vote_average", 0)/10
            }
    
    # Normalize scores
    df = pd.DataFrame.from_dict(all_movies, orient="index")
    df["content_score"] = (df["content_score"] - df["content_score"].min()) / 
                         (df["content_score"].max() - df["content_score"].min())
    df["collab_score"] = (df["collab_score"] - df["collab_score"].min()) / 
                        (df["collab_score"].max() - df["collab_score"].min())
    
    # Hybrid scoring
    df["hybrid_score"] = alpha * df["content_score"] + (1-alpha) * df["collab_score"]
    recommendations = df.sort_values("hybrid_score", ascending=False).head(6)
    
    return recommendations, target_movie["title"], target_movie.get("poster_path")

# --- Streamlit UI ---
col1, col2 = st.columns([3, 1])
with col1:
    movie_query = st.text_input("Enter a movie you like:", help="Try 'Inception' or 'The Dark Knight'")
with col2:
    alpha = st.slider("Algorithm balance:", 0.0, 1.0, 0.5, 
                     help="Left: More collaborative filtering | Right: More content-based")

if movie_query:
    with st.spinner("Finding recommendations..."):
        recommendations, target_title, target_poster = hybrid_recommend(movie_query, alpha)
    
    if not recommendations.empty:
        st.subheader(f"Because you liked: {target_title}")
        if target_poster:
            st.image(f"{POSTER_URL}{target_poster}", width=200)
        
        st.subheader("Recommended Movies")
        cols = st.columns(3)
        for idx, (movie_id, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                    if row["poster"]:
                        st.image(f"{POSTER_URL}{row['poster']}", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x450?text=No+Poster", 
                                use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.progress(row["hybrid_score"])
                    st.caption(f"Content match: {row['content_score']:.1%}")
                    st.caption(f"User preference: {row['collab_score']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No recommendations found. Try another movie.")
else:
    st.info("Enter a movie title to get recommendations")

st.markdown("---")
st.caption("Note: Collaborative filtering uses TMDB's 'similar movies' as a proxy for real user behavior data")
