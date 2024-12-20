# Movie Recommendation System

This project implements a movie recommendation system that suggests movies based on their similarity to a given input movie. The system leverages content-based filtering, using features like genres, directors, and top-billed actors to compute similarities between movies. The recommendations are enriched by displaying movie posters to provide a visually engaging experience.

---

## Features
1. **Data Loading and Merging**: Combines multiple datasets to create a comprehensive dataset for analysis.
2. **Data Cleaning and Feature Engineering**: Prepares the dataset by handling missing values and creating a composite feature for similarity computation.
3. **Similarity Matrix Computation**: Uses natural language processing techniques to compute movie similarities.
4. **Recommendation Function**: Recommends movies based on similarity scores.
5. **Poster Display**: Enhances the user experience by displaying posters of recommended movies.
6. **Example Execution**: Demonstrates the complete workflow of the recommendation system.

---

## Execution Steps

### 1. Load and Merge Data

This step involves loading the datasets and merging them to form a unified DataFrame for analysis.

- **Datasets Used**:
  - `Movies.csv`: Contains basic movie information.
  - `FilmDetails.csv`: Includes details like directors and top-billed actors.
  - `MoreInfo.csv`: Provides additional information such as runtime, budget, and revenue.
  - `PosterPath.csv`: Contains URLs to movie posters.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Image, display

# Load datasets
movies = pd.read_csv("Movies.csv")
film_details = pd.read_csv("FilmDetails.csv")
more_info = pd.read_csv("MoreInfo.csv")
poster_path = pd.read_csv("PosterPath.csv")

# Merge datasets
data = pd.merge(movies, film_details, on="id", how="left")
data = pd.merge(data, more_info, on="id", how="left")
data = pd.merge(data, poster_path, on="id", how="left")
```

---

### 2. Data Cleaning and Feature Engineering

The data is cleaned to handle missing values and engineered to create a composite feature for similarity computation.

```python
# Fill missing values
data["genres"] = data["genres"].fillna("Unknown")
data["director"] = data["director"].fillna("Unknown")
data["top_billed"] = data["top_billed"].fillna("Unknown")
data["user_score"] = data["user_score"].fillna(0)

# Combine relevant columns into a single string for feature extraction
data["features"] = data["genres"] + " " + data["director"] + " " + data["top_billed"]
```

---

### 3. Build Similarity Matrix

This step computes a similarity matrix using the combined features column. It transforms text into numerical form and calculates pairwise similarities.

```python
# Vectorize the combined features using CountVectorizer
vectorizer = CountVectorizer().fit_transform(data["features"])
similarity_matrix = cosine_similarity(vectorizer)
```

---

### 4. Recommendation Function

The recommendation function identifies movies similar to the input movie based on the similarity matrix.

```python
def recommend(movie_title, data, similarity_matrix):
    """
    Recommend movies based on similarity to the input movie title.

    Parameters:
    - movie_title (str): Title of the movie to base recommendations on.
    - data (DataFrame): The dataset containing movie information.
    - similarity_matrix (ndarray): Precomputed similarity matrix.

    Returns:
    - list: A list of recommended movie titles.
    """
    if movie_title not in data["title"].values:
        return ["Movie title not found in dataset."]

    idx = data[data['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_movies = [data.iloc[i[0]]['title'] for i in sorted_scores[1:6]]
    return recommended_movies
```

---

### 5. Display Posters

This function visually enhances recommendations by displaying movie posters.

```python
def show_posters(movie_list, data):
    """
    Display posters of recommended movies.

    Parameters:
    - movie_list (list): List of recommended movie titles.
    - data (DataFrame): Dataset containing movie information and poster links.
    """
    for movie in movie_list:
        poster_url = data[data['title'] == movie]['poster_path'].values[0]
        if poster_url:
            display(Image(url=poster_url))
        else:
            print(f"Poster not available for {movie}.")
```

---

### 6. Example Execution

An end-to-end demonstration of the movie recommendation system.

```python
movie_to_recommend = "Inception"  # Replace with a valid movie title from your dataset
recommended_movies = recommend(movie_to_recommend, data, similarity_matrix)

print("Recommendations for:", movie_to_recommend)
print(recommended_movies)

# Display posters of recommended movies
show_posters(recommended_movies, data)
```

---

## Conclusion
This project demonstrates a content-based movie recommendation system with features for data preparation, similarity computation, and visual engagement through movie posters. It is an excellent starting point for exploring recommendation systems and their applications.

