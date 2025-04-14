def hybrid_recommend(movie_title, alpha=0.5):
    """Hybrid recommendation combining content and collaborative filtering"""
    # Step 1: Find the target movie
    search_url = f"{BASE_URL}/search/movie"
    search_params = {"api_key": API_KEY, "query": movie_title}
    search_results = requests.get(search_url, params=search_params).json().get("results", [])
    
    if not search_results:
        return pd.DataFrame(), None, None
    
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
    
    # Normalize scores (corrected lines)
    df = pd.DataFrame.from_dict(all_movies, orient="index")
    df["content_score"] = ((df["content_score"] - df["content_score"].min()) / 
                         (df["content_score"].max() - df["content_score"].min()))
    df["collab_score"] = ((df["collab_score"] - df["collab_score"].min()) / 
                        (df["collab_score"].max() - df["collab_score"].min()))
    
    # Hybrid scoring
    df["hybrid_score"] = alpha * df["content_score"] + (1-alpha) * df["collab_score"]
    recommendations = df.sort_values("hybrid_score", ascending=False).head(6)
    
    return recommendations, target_movie["title"], target_movie.get("poster_path")
