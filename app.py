import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# --- Collaborative Filtering (Pandas-based) ---
def get_collaborative_recommendations(user_id):
    # Create user-item matrix
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Check if user ID exists
    if user_id not in user_movie_matrix.index:
        return []

    # Compute similarity of the input user with all others
    user_similarity = cosine_similarity([user_movie_matrix.loc[user_id]], user_movie_matrix)[0]
    
    # Weighted sum of ratings from similar users
    sim_scores = pd.Series(user_similarity, index=user_movie_matrix.index)
    weighted_ratings = user_movie_matrix.T.dot(sim_scores) / sim_scores.sum()
    
    # Recommend top 10 movies the user hasn't rated
    already_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    recommendations = weighted_ratings.drop(index=already_rated).sort_values(ascending=False).head(10)
    
    recommended_titles = movies[movies['movieId'].isin(recommendations.index)]['title'].tolist()
    return recommended_titles

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
    user_id = st.number_input("Enter your User ID (1–943):", min_value=1, max_value=943, step=1)
    if user_id:
        recommendations = get_collaborative_recommendations(user_id)
        if recommendations:
            st.subheader("Top Recommendations for You:")
            for i, movie in enumerate(recommendations):
                st.write(f"{i+1}. {movie}")
        else:
            st.warning("No recommendations available for this user.")
