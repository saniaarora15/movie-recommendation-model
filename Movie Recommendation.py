
import numpy as np
import pandas as pd
import warnings

#if we get any warning we can ignore them
warnings.filterwarnings('ignore')

#Column names
col=["user_id","item_id","rating","timestamp"]

#sep is the character separating column from another column
data=pd.read_csv("u.data",sep='\t',names=col)

#viewing first 5 rows of the dataset
data.head()

#Dimenstions of the dataset
data.shape

#number of unique users
data['user_id'].nunique()

#Checking for null values
print(data.isnull().sum())

#number of unique movies
data['item_id'].nunique()

#reading movie rating data
item= pd.read_csv('u.item',sep='\|',header=None)
item.head()

item.shape

#1st and 2nd column
movie_title= item[[0,1]]

#we require only 1st and 2nd col from movie rating(item) dataframe
# Naming the columns in the movie_title dataframe
movie_title.columns=['item_id','Name']

movie_title.head()

# Merging data and movie_title dataframes along the common column "item_id"
movie=pd.merge(data,movie_title,on="item_id")

movie.shape

movie.head(10)


## Exploratory data analysis


import matplotlib.pyplot as plt
import seaborn as sns
#data visualization modules

# grouping by names to view average ratings of each movie in the dataset
movie.groupby('Name')['rating'].mean().sort_values(ascending=False)
#here the 5 rated movie may not be liked by everyone as it is possible only a few people(like only 1) rated the movie

#how many people rated each movie
movie.groupby('Name')["user_id"].count().sort_values(ascending=False)

#Marlene Dietrich maybe or may not be liked by the user
#it has a rating of 5 but only 1 person has rated it.

# creating a dataframe to store average rating of movie
Ratings=pd.DataFrame(movie.groupby('Name')['rating'].mean())

Ratings.head()

movie.head()

# Adding a column "Rating Count" to Ratings to store the number of ratings each movie has
Ratings['Rating count']= pd.DataFrame(movie.groupby('Name')["user_id"].count())

Ratings.head()

# Sorting the 'Ratings' dataframe in descending order of "Rating counts"
Ratings.sort_values(by='Rating count',ascending=False)

#Visualising rating count

plt.figure(figsize=(10,6))
# 10 is width and 6 is height of the figure
plt.style.use('seaborn')
plt.xlabel("Rating Count")
plt.ylabel("frequency")
plt.title("Histogram of number of ratings per movie")
plt.hist(Ratings['Rating count'],bins=70)
plt.show()

#there are only a few movies rated by 100+ people. Most of the movies here are rated by less than 100 people.
#about 500 movies are only rated by 0-10 people only.

plt.hist(Ratings['rating'],bins=70)
plt.show()

sns.jointplot(x='rating', y='Rating count',data=Ratings, alpha=0.5)

#ratings increase, no. of ratings increase
#except 5
#ratings are max for 3-4 rated movies


# # Creating Movie recommendation Model

#we are recommending movies based on what other people liked
#good rated movies are suggested to watch

#matrix rows-users,columns-movie names
movie.head()

movmatrix=movie.pivot_table(index='user_id',columns='Name',values='rating')
#this is similar to
#movie.groupby(['user_id','Name'])['rating'].mean().unstack()

print(movmatrix)

movie.groupby(['user_id','Name'])['rating'].mean()
#This gives the movie rated by a user
#mean(if user rated a movie more than once, we want the mean rating by the user)

#movie.groupby(['user_id','Name'])['rating'].mean().unstack()
#unstack shows the hidden NaN values and data in a tabular form.

# Sorting 'Ratings' column by 'rating' and 'Rating Count'
a=pd.DataFrame(Ratings.sort_values(by=['rating','Rating count'],ascending=False))
a
#in terms of ratings and number of ratings, star wars is at the top.

# A list of all the movies
list(movie['Name'].unique())

#set(movie['Name'])

# Just for refernce, Toy Story(1995) is coorelated with every other movie
Toy_story_rating=movmatrix['Toy Story (1995)']
Toy_story_rating

# we can correlate this series with the whole movie matrix. Basis of the highest correlation with the movie chosen by the user,
# movie would be suggested.
Toy_sim=movmatrix.corrwith(Toy_story_rating)
Toy_sim

corr_ToyStory=pd.DataFrame(Toy_sim,columns=['Correlation'])
corr_ToyStory
#NaN values because no user rated toy story and other movie so NaN.
#correlation between the 2 doesn't exist

# the dataframe is showing correlation of each movie with Toy story.

corr_ToyStory.dropna(inplace=True)
#inplace to alter the original data frame and not create a copy with changes

corr_ToyStory

corr_ToyStory.sort_values('Correlation',ascending=False)

# correlation is on the basis of users rating to Toy Story and similar rating given by the same users to another movie

#in order to avoid uncertainty, we will have only those movies which are rated by more then 100 people.
#a movie cannot be perfectly correlated with another movie.

Ratings.head()

corr_ToyStory

corr_ToyStory=corr_ToyStory.join(Ratings['Rating count'])

corr_ToyStory[corr_ToyStory['Rating count']>100].sort_values('Correlation',ascending=False)

movie_name=movie['Name']
movie_name


# ## Predict Function

def predict(movie_name):
    movie_rating=movmatrix[movie_name]
    movie_similarity=movmatrix.corrwith(movie_rating)
    corr_movie=pd.DataFrame(movie_similarity,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(Ratings['Rating count'])
    predictions=corr_movie[corr_movie['Rating count']>100].sort_values('Correlation',ascending=False)
    return predictions


predict('Star Wars (1977)')

