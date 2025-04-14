import streamlit as st
import pickle
import requests

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

API_KEY = 'd52a6c7f0a93a3293e13015b2e41915b'

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    return "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/300x450?text=No+Image"

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_titles = []
    recommended_posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_titles.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_titles, recommended_posters

st.set_page_config(page_title="🎥 Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("## Select a movie you like and discover similar ones below")

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    st.markdown("---")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.markdown(f"**{names[i]}**")
