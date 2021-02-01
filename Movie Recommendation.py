#!/usr/bin/env python
# coding: utf-8

# Content based (similarity based)
# 

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


#if we get any warning we can ignore them
warnings.filterwarnings('ignore')


# In[3]:


#Column names
col=["user_id","item_id","rating","timestamp"]
#sep is the character separating column from another column
data=pd.read_csv("u.data",sep='\t',names=col)


# In[4]:


data.head()


# In[5]:


data.shape


# In[7]:


#number of unique users
data['user_id'].nunique()


# In[8]:


print(data.isnull().sum())


# In[9]:


#checking for null values


# In[10]:


#number of unique movies
data['item_id'].nunique()


# In[12]:


#reading movie rating data
item= pd.read_csv('u.item',sep='\|',header=None)
item.head()


# In[13]:


item.shape


# In[14]:


#1st and 2nd column
movie_title= item[[0,1]]


# In[15]:


#we require only 1st and 2nd col from movie rating(item) dataframe
movie_title.columns=['item_id','Name']


# In[16]:


movie_title.head()


# In[17]:


movie=pd.merge(data,movie_title,on="item_id")


# In[26]:


movie.shape


# In[18]:


movie.head(10)


# # Exploratory data analysis

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
#data visualization modules


# In[20]:


movie.groupby('Name')['rating'].mean().sort_values(ascending=False)
#here the 5 rated movie may not be liked by everyone as it is possible only a few people(like only 1) rated the movie


# In[21]:


#how many people rated each movie
movie.groupby('Name')["user_id"].count().sort_values(ascending=False)


# In[22]:


#Marlene Dietrich maybe or may not be liked by the user
#it has a rating of 5 but only 1 person has rated it.


# In[23]:


Ratings=pd.DataFrame(movie.groupby('Name')['rating'].mean())


# In[24]:


Ratings.head()


# In[25]:


movie.head()


# In[30]:


movie.head()


# In[28]:


Ratings['Rating count']= pd.DataFrame(movie.groupby('Name')["user_id"].count())


# In[29]:


Ratings.head()


# In[88]:


Ratings.sort_values(by='Rating count',ascending=False)


# In[38]:


plt.figure(figsize=(10,6))
#10 is width and 6 is height of the figure
plt.style.use('seaborn')
plt.xlabel("Rating Count")
plt.ylabel("frequency")
plt.title("Histogram of number of ratings per movie")
plt.hist(Ratings['Rating count'],bins=70)
plt.show()


# In[ ]:


#there are only a few movies rated by 100+ people. Most of the movies here are rated by less than 100 people.
#about 500 movies are only rated by 0-10 people only.


# In[108]:


plt.hist(Ratings['rating'],bins=70)
plt.show()


# In[109]:


sns.jointplot(x='rating', y='Rating count',data=Ratings, alpha=0.5)


# In[110]:


#ratings increase, no. of ratings increase
#except 5
#ratings are max for 3-4 rated movies


# # Creating Movie recommendation Model

# In[112]:


#we are recommending movies based on what other people liked
#good rated movies are suggested to watch


# In[47]:


#matrix rows-users,columns-movie names
movie.head()


# In[39]:


movmatrix=movie.pivot_table(index='user_id',columns='Name',values='rating')
#this is similar to
#movie.groupby(['user_id','Name'])['rating'].mean().unstack()


# In[40]:


print(movmatrix)


# In[45]:


movie.groupby(['user_id','Name'])['rating'].mean()
#This gives the movie rated by a user
#mean(if user rated a movie more than once, we want the mean rating by the user)


# In[43]:


movie.groupby(['user_id','Name'])['rating'].mean().unstack()
#unstack shows the hidden NaN values and data in a tabular form.


# In[57]:


a=pd.DataFrame(Ratings.sort_values(by=['rating','Rating count'],ascending=False))
a


# In[58]:


#in terms of ratinga and number of ratings, star wars is at the top.


# In[85]:


list(movie['Name'].unique())


# In[91]:


#set(movie['Name'])


# In[125]:


Toy_story_rating=movmatrix['Toy Story (1995)']
Toy_story_rating


# In[ ]:


# we can correlate this series with the whole movie matrix. Basis of the highest correlation with the movie chosen by the user,
# movie would be suggested.


# In[122]:


Toy_sim=movmatrix.corrwith(Toy_story_rating)
Toy_sim


# In[123]:


corr_ToyStory=pd.DataFrame(Toy_sim,columns=['Correlation'])
corr_ToyStory
#NaN values because no user rated toy story and other movie so NaN.
#correlation between the 2 doesn't exist


# In[97]:


# the dataframe is showing correlation of each movie with Toy story.


# In[103]:


corr_ToyStory.dropna(inplace=True)
#inplace to alter the original data frame and not create a copy with changes


# In[104]:


corr_ToyStory


# In[108]:


corr_ToyStory.sort_values('Correlation',ascending=False)


# In[109]:


# correlation is on the basis of users rating to Toy Story and similar rating given by the same users to another movie


# In[111]:


#in order to avoid uncertainty, we will have only those movies which are rated by more then 100 people.
#a movie cannot be perfectly correlated with another movie.


# In[112]:


Ratings.head()


# In[113]:


corr_ToyStory


# In[115]:


corr_ToyStory=corr_ToyStory.join(Ratings['Rating count'])


# In[118]:


corr_ToyStory[corr_ToyStory['Rating count']>100].sort_values('Correlation',ascending=False)


# In[121]:


movie_name=movie['Name']
movie_name


# ## Predict Function

# In[126]:


def predict(movie_name):
    movie_rating=movmatrix[movie_name]
    movie_similarity=movmatrix.corrwith(movie_rating)
    corr_movie=pd.DataFrame(movie_similarity,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(Ratings['Rating count'])
    predictions=corr_movie[corr_movie['Rating count']>100].sort_values('Correlation',ascending=False)
    return predictions


# In[127]:


predict('Star Wars (1977)')


# In[ ]:




