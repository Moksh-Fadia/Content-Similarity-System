# Movie Content Similarity System

A content-based **Movie Recommendation System** built using **Flask** and **Sentence Transformers** that recommends similar movies based on semantic similarity.

The system uses movie metadata such as overview, genre, cast, and title to generate semantic embeddings and uses cosine similarity to identify similar movies.

It also stores recent searches using **SQLite** and displays them dynamically on the results page.

---

## Features

- Search for a movie and receive the top 5 similar recommendations.
- Semantic similarity using SentenceTransformer embeddings.
- Uses movie overview, genre, cast, and title as recommendation features.
- Genre and cast are given additional weight during feature construction.
- Recent search history stored using SQLite.
- Dynamic results rendered using Flask and Jinja2.
- Deployed publicly using Gunicorn and Render.

---

## Performance Optimization

### Persistent Movie Embeddings

Generating SentenceTransformer embeddings for the entire movie dataset during every application startup was expensive.

To avoid repeated model inference:

- Movie embeddings are generated once.
- The embeddings are persisted as `data/movie_embeddings.npy`.
- Subsequent startups load the precomputed embeddings using NumPy.

This reduced the measured embedding preparation/startup step from approximately **63 seconds to 0.24 seconds**.

### Memory Optimization

The initial implementation precomputed a full pairwise cosine similarity matrix for all movies.

For approximately 10,000 movies, this required a 10,000 × 10,000 similarity matrix containing approximately 100 million similarity values.

This caused the application to exceed Render's 512 MiB memory limit.

The final implementation avoids storing this full matrix.

Instead, cosine similarity is calculated on demand between the selected movie's embedding and all movie embeddings. This significantly reduces memory usage while preserving the recommendation logic.


### Performance Measurement

End-to-end request latency was measured locally, including:

- Input validation
- SQLite database operations
- Recommendation computation
- Similarity ranking
- Genre filtering
- Recent search retrieval
- Jinja template rendering

Across 17 local test requests:

- Average response time: ~14 ms
- Median response time: ~4.3 ms

### Tech Stack

- Backend:
Python, Flask, Gunicorn, SQLite

- ML / NLP:
SentenceTransformers (all-MiniLM-L6-v2), Scikit-learn

- Data Processing:
Pandas, NumPy

- Frontend:
HTML, CSS, Jinja2

- Deployment:
Render


### How It Works

1) The user enters a movie title through the Flask web interface.
2) Movie metadata is cleaned and combined from:
Overview, Genre,
Cast,
Movie title

3) Genre information is given 3x weight and cast information 2x weight when constructing the combined text representation.
4) The combined movie metadata is converted into a 384-dimensional semantic embedding using:
all-MiniLM-L6-v2
5) When a user searches for a movie, the system calculates cosine similarity between the selected movie's embedding and all stored movie embeddings.
6) Similarity scores are sorted in descending order and the top 5 recommendations are returned.
7) SQLite stores the user's recent searches and timestamps.
8) Flask and Jinja2 dynamically render the recommendations and recent search history.


### Database

The application uses SQLite to store recent searches in:

app/search_history.db

The search_history table contains:

id, Searched movie title, Search timestamp


### Installation

Clone the repository and install the dependencies:

pip install -r requirements.txt
Run Locally

Start the Flask application:

python main.py

Then open:

http://127.0.0.1:5000/search

The root route redirects to the movie search page.


### Deployment

The application is deployed using Gunicorn on Render.
Try it out here: https://content-similarity-system-wynv.onrender.com