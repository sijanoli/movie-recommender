import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "d52a6c7f0a93a3293e13015b2e41915b"
BASE_URL = "https://api.themoviedb.org/3"

def search_movie(query):
    url = f"{BASE_URL}/search/movie"
    params = {"api_key": API_KEY, "query": query}
    res = requests.get(url, params=params)
    return res.json().get("results", [])

def fetch_overview(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}
    res = requests.get(url, params=params)
    return res.json().get("overview", ""), res.json().get("poster_path", "")

def fetch_similar_movies(base_movie):
    results = search_movie(base_movie)
    if not results:
        return [], base_movie

    base = results[0]
    base_id = base['id']
    base_title = base['title']
    base_overview, base_poster = fetch_overview(base_id)

    candidates = results[:10]
    all_titles = []
    all_overviews = []
    posters = []

    for m in candidates:
        overview, poster_path = fetch_overview(m['id'])
        if overview:
            all_titles.append(m['title'])
            all_overviews.append(overview)
            posters.append(f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None)

    # TF-IDF similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_overviews)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    base_idx = 0
    sim_scores = list(enumerate(cosine_sim[base_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended = [(all_titles[i], posters[i]) for i, _ in sim_scores]

    return recommended, base_title

# Streamlit UI
st.set_page_config(page_title="🎥 Real-Time Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Real-Time Movie Recommender (TMDb)</h1>", unsafe_allow_html=True)
movie_input = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recs, base = fetch_similar_movies(movie_input)
        if recs:
            st.success(f"Recommendations based on **{base}**")
            cols = st.columns(5)
            for i, (title, poster) in enumerate(recs):
                with cols[i]:
                    if poster:
                        st.image(poster, use_column_width=True)
                    st.caption(title)
        else:
            st.warning("No recommendations found. Try a different movie.")
