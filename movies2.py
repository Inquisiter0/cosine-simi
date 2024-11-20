from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__, template_folder="movies_templates", static_folder="movies_templates/static")

# Load datasets
movies_path = 'datasets/movies.csv'  
movies = pd.read_csv(movies_path)
# Limit the dataset to a smaller subset for testing
movies = pd.read_csv(movies_path).head(10000)  # Adjust the number based on available memory

# Handle missing or empty genres and prepare data for TF-IDF
movies['genres'] = movies['genres'].fillna('').apply(lambda x: ' '.join(x.split('|')))

# Prepare TF-IDF for genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Content-based recommendation function
def content_based_recommend(movie_id, n_recommendations=10):
    if movie_id < 0 or movie_id >= len(movies):
        return ["Invalid movie ID. Please enter a valid ID."]
    
    similar_scores = cosine_sim[movie_id]
    similar_movies = similar_scores.argsort()[-n_recommendations-1:][::-1]
    recommended_movie_titles = movies.iloc[similar_movies]['title'].tolist()
    return recommended_movie_titles

# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    movie_id = None
    movie_title = None
    if request.method == 'POST':
        try:
            movie_id = int(request.form['movie_id']) - 1  # Convert to zero-based index
            movie_title = movies.iloc[movie_id]['title']  # Get the title for the entered movie ID
            recommendations = content_based_recommend(movie_id)
            return render_template('content_recommend.html', movie_id=movie_id+1 , movie_title=movie_title, recommendations=recommendations)
        except Exception as e:
            return f"Error: {e}", 400

    return render_template('content_recommend.html', movie_id=movie_id, movie_title=movie_title, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
