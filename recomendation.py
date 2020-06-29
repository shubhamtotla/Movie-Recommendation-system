import pandas as pd
import numpy as np
#import datasets
credits = pd.read_csv("C:\Users\SHUBHAM TOTLA\Desktop\Data analytics\movie dataset\credits.csv",engine='python')
keywords=pd.read_csv("C:\Users\SHUBHAM TOTLA\Desktop\Data analytics\movie dataset\keywords.csv",engine='python')
metadata=pd.read_csv("C:\Users\SHUBHAM TOTLA\Desktop\Data analytics\movie dataset\movies_metadata.csv",engine='python')
ratings = pd.read_csv("C:/Users/SHUBHAM TOTLA/Desktop/Data analytics/movie dataset/ratings_small.csv",engine='python')
#Recommendng movies based on plot
metadata.shape
#Remove all the movies that are less than 90 percentile of vote count
m = metadata['vote_count'].quantile(0.94)
# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape
#reset index
q_movies = q_movies.reset_index(drop = True)
q_movies.index

#Recommend based on overview


q_movies['overview'].head(10)
from sklearn.feature_extraction.text import TfidfVectorizer
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN with an empty string
q_movies['overview'] = q_movies['overview'].fillna('')
#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])
#Output the shape of tfidf_matrix
tfidf_matrix.shape
#Compute similarity score by importing linear_kernel
from sklearn.metrics.pairwise import linear_kernel
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Generate title as index 
indices = pd.Series(q_movies.index,index=q_movies['title']).drop_duplicates()
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie and convert it into a list
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores from second element since first one is itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies leaving first element
    sim_scores = sim_scores[1:11]

   # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    print(q_movies.iloc[movie_indices][['title']])
    
#get_recommendations('The Dark Knight Rises')
    

#Recommendation based on actors,genres,keywords etc


# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
q_movies['id'] = q_movies['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
q_movies = q_movies.merge(credits, on='id')
q_movies = q_movies.merge(keywords, on='id')

q_movies.shape

# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(literal_eval)
    
# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
    
# Returns the list of top 3 actors  
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 actors exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
    
#get directors using the function and create its respective column
q_movies['director']=q_movies['crew'].apply(get_director)
#get actors using the function and create its respective column
q_movies['actors']=q_movies['cast'].apply(get_list)
#get keywords using the function and create its respective column
q_movies['keywords']=q_movies['keywords'].apply(get_list)
#get genres using the function and create its respective column
q_movies['genres']=q_movies['genres'].apply(get_list)

# Print the new features of the first 3 films
q_movies[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
# Apply clean_data function to your features.
q_movies['director']=q_movies['director'].apply(clean_data)
q_movies['actors']=q_movies['actors'].apply(clean_data)
q_movies['keywords']=q_movies['keywords'].apply(clean_data)
q_movies['genres']=q_movies['genres'].apply(clean_data)
q_movies[['title', 'actors', 'director', 'keywords', 'genres']].head(3)
#Create a new row that contains all the above inf
def create_row(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['actors']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

q_movies['new'] = q_movies.apply(create_row, axis=1)

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(decode_error='ignore')
count_matrix = count.fit_transform(q_movies['new'])
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
#Reset index and generate title as index
q_movies = q_movies.reset_index(drop = True)
indices = pd.Series(q_movies.index, index=q_movies['title'])
#Recommend movies using the function
#get_recommendations('Pulp Fiction', cosine_sim2)


#Improved Recommendations


# Calculate C
C = metadata['vote_average'].mean()
# Calculate the minimum number of votes required to be in the chart, m i.e., q_movies
m = metadata['vote_count'].quantile(0.94)
# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
def improved_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie and convert it into a list
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores from second element since first one is itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies leaving first element
    sim_scores = sim_scores[1:11]

   # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    movies = q_movies.iloc[movie_indices][['title', 'score']]
    movies = movies.sort('score', ascending=False)
    print(movies.head(10))
#improved_recommendations('The Dark Knight Rises')





