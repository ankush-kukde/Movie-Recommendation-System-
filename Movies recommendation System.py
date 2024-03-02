#Stepwise making of this code is on my github , here is the link:

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer   #Used to convert characteral data into numerical data.
from sklearn.metrics.pairwise import cosine_similarity        #Gives similarity score of that particular movie with other movies.


#Get the Data 
movies_data=pd.read_csv(r'C:\Users\hkukd\OneDrive\Documents\Python Codes\movies.csv')

nrows=movies_data.shape[0]
ncoloumns=movies_data.shape[1]

selected_features=['genres','keywords','tagline','cast','director']

#replacing the null values with null strings for above selected genres
for features in selected_features:
    movies_data[features]=movies_data[features].fillna('')

#combining all the 5 features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

#converting the text data to feature vector
vectorizer= TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)

#getting the similarity scores using cosine similarity
similarity=cosine_similarity(feature_vectors)

#Asking the movie name
movie_name=input('Enter the name of movie:')

#creating a list of names of all movies given in database
list_of_all_titles=movies_data['title'].tolist()

#finding the cloasest match for the name of movie
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)

close_match=find_close_match[0]
index_of_movie=movies_data[movies_data.title==close_match]['index'].values[0]

#finding similar values based on ondex value
similarity_score=list(enumerate(similarity[index_of_movie]))

#sorting the movies based on their similarityscore
sorted_similarity_movies=sorted(similarity_score , key=lambda x:x[1], reverse=True)

#final printing of titles of most similalar movies using their index

print('Movies suggested for you: \n')

i=1
for movie in sorted_similarity_movies:
  index=movie[0]
  title_from_index=movies_data[movies_data.index==index]['title'].values[0]
  if(i<30):
    print(i,'.',title_from_index)
    i+=1


