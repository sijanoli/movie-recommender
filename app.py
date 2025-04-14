import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="MovieMagic AI", page_icon="🎬", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .movie-title {
        font-size: 18px !important;
        font-weight: bold !important;
        margin-top: 10px !important;
        text-align: center;
    }
    .recommendation-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #0E1117;
        margin: 10px 0;
        transition: transform 0.2s;
        height: 100%;
    }
    .recommendation-card:hover {
        transform: scale(1.02);
        background-color: #1a1d25;
    }
    .header {
        color: #FF4B4B;
    }
    .stSlider>div>div>div>div {
        background-color: #FF4B4B;
    }
    .search-result {
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

def search_movie(title):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

def hybrid_recommend(title, alpha=0.5):
    results = search_movie(title)
    if not results:
        return [], None, None

    target = results[0]
    all_movies = results[:10]

    titles = [m['title'] for m in all_movies]
    overviews = [m.get('overview', '') for m in all_movies]
    votes = [m.get('vote_average', 0) for m in all_movies]
    poster_paths = [m.get('poster_path', '') for m in all_movies]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(overviews)
    content_sim = cosine_similarity(tfidf_matrix)

    idx = 0
    content_scores = content_sim[idx]

    vote_scores = pd.Series(votes).fillna(0)
    vote_scores = (vote_scores - vote_scores.min()) / (vote_scores.max() - vote_scores.min())

    final_scores = alpha * content_scores + (1 - alpha) * vote_scores
    sorted_idx = final_scores.argsort()[::-1][1:6]

    recommendations = [(titles[i], poster_paths[i], votes[i]) for i in sorted_idx]
    return recommendations, target['title'], target.get('poster_path', '')

# --- Streamlit Interface ---
st.title("🎬 MovieMagic AI")
st.markdown("### Discover Your Next Favorite Movie")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    movie_title = st.text_input("Enter a movie you love:", placeholder="The Dark Knight, Inception...")
with col2:
    alpha = st.slider("Content vs Popularity", 0.0, 1.0, 0.5, help="Slide left for popular movies, right for similar content")

if movie_title:
    with st.spinner('🔍 Finding recommendations...'):
        recommended, match_title, match_poster = hybrid_recommend(movie_title, alpha=alpha)
        
    if recommended:
        # Display search result and recommendations together
        st.markdown(f"### Because you liked: **{match_title}**")
        
        # Create columns for the search result and recommendations
        col1, spacer, col2 = st.columns([2, 0.2, 8])
        
        with col1:
            if match_poster:
                st.image(POSTER_BASE_URL + match_poster, use_container_width=True, caption=match_title)
            else:
                st.image("https://via.placeholder.com/150x225?text=No+Poster", use_container_width=True, caption=match_title)
        
        with col2:
            st.markdown("### Recommended Movies")
            
            # Display recommendations in a grid
            cols = st.columns(5)
            for idx, (rec_title, poster_path, vote) in enumerate(recommended):
                with cols[idx % 5]:
                    with st.container():
                        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                        if poster_path:
                            st.image(POSTER_BASE_URL + poster_path, use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/150x225?text=No+Poster", use_container_width=True)
                        st.markdown(f'<p class="movie-title">{rec_title}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="text-align: center;">⭐ {vote}/10</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No recommendations found. Try a different movie title.")
else:
    st.info("✨ Enter a movie title to get personalized recommendations")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Powered by TMDB API and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
