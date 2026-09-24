# define API endpoints for Flask (here /recommend)

from flask import Blueprint, request, render_template, redirect   # Blueprint groups related routes together, request lets us read incoming HTTP request data, render_template renders HTML files from templates folders 
from app.recommender import MovieRecommender   # import the recommender class so that it can be called inside API routes
from app.db import add_search, get_recent_searches, init_db
import time

# creates a Blueprint named 'routes' to group routes
bp = Blueprint('routes', __name__, template_folder="templates")  # __name__ tells Flask where this blueprint is defined (ie. current file)

# creates one recommender instance when the server starts;
# loads the movie data, preprocesses it, and loads precomputed embeddings
recommender = MovieRecommender("data/imdb_movies.csv")

init_db()

@bp.route("/")
def home():
    return redirect("/search")

# define/declare the route /search to show HTML form
@bp.route("/search", methods=["GET"])   # this route will respond only to GET requests; loads the html input page
# the decorator tells Flask that when someone sends a GET request to /search run the function below

def search():   
    return render_template("search.html")    # Finds the HTML file, Processes it, Sends it as the HTML response to the browser


# Route to handle form submission
@bp.route("/recommend-form", methods=["POST"])   # new route responds to POST requests (from the html form); this route is where the HTML form sends its data

def recommend_form():
    start_time = time.time()   # start timer to measure how long the recommendation process takes (for performance monitoring)

    try:
        title = request.form.get("title", "").strip()   # reads the "title" field from the submitted form data; remove extra spaces before/after the title

        if not any(char.isalnum() for char in title):   # checks if the title field contains any special characters (no letters or numbers)
            return render_template("results.html", error="Movie title must contain letters or numbers", title=title)
        
        add_search(title)  # adds the search to the sqlite db

        result = recommender.recommend(title, num_recommendations=5)    # call the recommend method of recommender instance to get movie recommendations for the given title
    # result returns a dictionary like: { "recommendations": ["Movie A", "Movie B", "Movie C"] } ie. top 5 similar movies        
        
        if "error" in result:   # checks if the dictionary result has a key named 'error'; eg: result = {"error": "Movie not found"}
            return render_template("results.html", error=result["error"], title=title)   # this title will be used in results.html in place of {{ title }}; that error msg in result (ie. result["error"]) will be used in results.html in place of {{ error }}
        
        history = get_recent_searches(limit=5)

        response = render_template("results.html", recommendations=result["recommendations"], title=title, history=history)

        end_time = time.time()   # end timer
        print(f"Total request time: {end_time - start_time:.4f} sec")

        return response
    
    except Exception as e:    # handles unexpected errors
        print(e)       
        return render_template("results.html", error="An unexpected error occurred", title=title)
    
    finally:
        end_time = time.time()
        print(f"Total request time: {end_time - start_time:.4f} sec")
    


