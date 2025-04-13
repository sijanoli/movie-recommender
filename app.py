import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# --- Content-Based Model ---
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(title):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# --- Collaborative Filtering (using Surprise) ---
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

def get_collaborative_recommendations(user_id):
    movie_ids = movies['movieId'].unique()
    predictions = [svd.predict(user_id, movie_id) for movie_id in movie_ids]
    top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
    top_movie_ids = [pred.iid for pred in top_preds]
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")
method = st.selectbox("Choose Recommendation Method:", ["Content-Based", "Collaborative Filtering"])

if method == "Content-Based":
    title = st.text_input("Enter a movie title:")
    if title:
        recommendations = get_content_recommendations(title)
        if recommendations:
            st.subheader("Top Recommendations:")
            for i, movie in enumerate(recommendations):
                st.write(f"{i+1}. {movie}")
        else:
            st.warning("Movie not found. Try typing full title.")

else:
    user_id = st.number_input("Enter your User ID (1–610):", min_value=1, max_value=610, step=1)
    if user_id:
        recommendations = get_collaborative_recommendations(user_id)
        st.subheader("Top Recommendations for You:")
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")
