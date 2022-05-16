#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ### Load movie data

# In[2]:


data=pd.read_csv(r'C:\Assignment\movie_metadata.csv')
data.head()


# In[3]:


#get the count of na values in each column
data.isna().sum()


# In[4]:


#replacing nan with 0 for analysis
data=data.replace(np.nan,0)
data.head()


# ### dropping duplicate records

# In[5]:


print(data.shape)
data=data.drop_duplicates()
print(data.shape)


# ### storing numerical and categorical columns in separate lists 

# In[113]:


cat_cols = list(data.select_dtypes(include='object').columns)
num_cols = list(data.select_dtypes(exclude='object').columns)


print('categorical columns are {}'.format(cat_cols))
print('continous columns are {}'.format(num_cols))


# ### plotting correlation matrix for numerical columns

# In[132]:


_, ax = plt.subplots(figsize=(12, 10))
#YlGnBu
#coolwarm
sns.heatmap(data[num_cols].corr(), vmin=-1, vmax=1, cmap='YlGnBu', annot=True, ax=ax)


# In[117]:


sns.pairplot(data[num_cols])


# <b>By observing above output , we can say there are null values present in whole data except for few columns like movie_id,imdb_score etc</b>

# ### Movie_Title column analysis and cleaning

# In[6]:


print("number of records present in dataframe are ",data.shape[0])
print("Total number of unique movie_titles in dataframe are ",len(data['movie_title'].unique()))
# len(data['movie_title'].unique())


# In[7]:


from collections import Counter
dict(Counter(data['movie_title']))
# data[['movie_title']]


# <b>After observing movie_title there are xa0 at end of the string</b>

# <b>replacing \xao with empty string</b>

# In[8]:


data[data['movie_title']=='Spider-Man 3\xa0']['movie_title']
data['movie_title']=data['movie_title'].apply(lambda x:x.replace(u'\xa0',u''))
data['movie_title']


# In[10]:


# t=data['color'].value_counts()
# t.values
# plt.plot(list(t.index),list(t.values))
# plt.xlabel('Color')
# plt.ylabel('movies')
# plt.title('Count of movies for unique color type')
# plt.show()
import seaborn as sns
sns.countplot(data=data,x='color')


# <b>From above result we can observe that majority of the movies are Color</b>

# In[11]:


len(data['director_name'].unique())


# <b>there are 2399 unique directors for 5000 movies</b>

# In[119]:


data['director_facebook_likes'].plot.hist(bins=20, log=True)


# In[70]:


# sns.countplot(data=data,x='director_name')
t=data['director_name'].value_counts()
t.values
plt.figure(figsize=(15,8))
plt.plot(list(t.index),list(t.values))
plt.xlabel('director_name')
plt.ylabel('count')
plt.title('Count of movies for unique color type')
plt.show()


# In[13]:


t


# <b>from above plot and results </b>
# <li>there are 103 records where director name is not given</li>
# <li>Steven Spielberg has directed most number of movies(26) followed by Woody Allen(22) in given dataset</li>

# ### num_critic_for_reviews

# In[15]:


data['num_critic_for_reviews']


# In[35]:


data[['num_critic_for_reviews']].agg([np.min,np.max,np.sum,np.mean,np.median,np.std,np.var])


# <b>
#     <li>From above results we can observe minimum value for num_critic_for_reviews is 0</li>
#     <li>mean value is 138</li>
# 
# </b>

# In[42]:



data[['duration']].describe()


# from above results we can observe that
# 75% of movies have duration less than or equal to 118 mins

# ## analysis on gross column

# In[44]:


# data['g']
data[['gross']].describe()


# In[46]:


data[data['gross']==max(data['gross'])][['director_name','movie_title','genres','language','country']]


# <b>Avatar movie has collected highest gross among all movies which was directed by James cameron</b>

# In[118]:


_, ax = plt.subplots(figsize=(18, 6))
sns.regplot(x=data['budget'].apply(np.log10), y=data['gross'], ax=ax)


# ### Genres 

# In[50]:


data['genres']


# <b>By observing above data we can see that for each movies there are multiple genres</b>

# ### Language

# In[55]:


data['language'].value_counts()


# By observing above data we can see that there are majority of movies in english followed by french

# In[62]:


data.groupby(['language'])['gross'].agg([np.min,np.max,np.mean]).reset_index().sort_values(by='amax',ascending=False).head(5)


# <b>
#     <li>By observing above results we can see that English Language films has collected highest gross</li>
#     <li>Avearge gross collections are very high for Mandarin language</li>
# </b>

# In[67]:


# sns.countplot(data,x='title_year')
sns.set(rc = {'figure.figsize':(15,8)})
sns.countplot(data=data,x='title_year')


# <b>
#     from above plot we can observe that as year is increasing ,number of movies release increased
# </b> 

# ### analysis on country column

# In[75]:


print("number of unique countries which are present in data are ",len(data['country'].unique()))
# len(data['country'].unique())


# In[121]:


_, ax = plt.subplots(figsize=(24, 6))
sns.barplot(x='country', y='gross', data=data, ax=ax, color='#30BA8B')
_ = plt.xticks(rotation=90)


# In[77]:


data.groupby(['country'])['gross'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='amax',ascending=False).head(5)


# <b>
#     <li>USA has collected maximum gross in total when compared to others</li>
#     <li>New Zealand has highest mean gross per movie when compared to others</li>
# </b>

# In[81]:


# data.columns
data.groupby(['country'])['budget'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='amax',ascending=False).head(5)


# <b>
#     <li>From above results we can see that South Korea movie has produced with high budget</li>
#     <li>Hungary has produced high mean budget movies</li>
# </b>

# In[87]:


# data.columns
data.groupby(['country'])['imdb_score'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='amax',ascending=False).head(5)


# <b>
#     <li>Movie which was directed by canadian director has highest imdb score</li>
# </b>

# ### Bucketing imdb score

# In[91]:


data.columns
def imdb_score_bucket(x):
    if x>=9:
        return '>9'
    elif (x>=8  and x<9):
        return '8-9'
    elif (x>=7  and x<8):
        return '7-8'
    elif (x>=6  and x<7):
        return '6-7'
    elif (x>=5  and x<6):
        return '5-6'
    elif (x>=4  and x<5):
        return '4-5'
    elif (x>=3  and x<4):
        return '3-4'
    elif (x>=2  and x<3):
        return '2-3'
    elif (x>=1  and x<2):
        return '1-2'
    else:
        return '<1'
data['imdb_score_bucket']=data['imdb_score'].apply(lambda x:imdb_score_bucket(x))


# In[94]:


# buckets=data['imdb_score_bucket'].value_counts()
sns.set(rc = {'figure.figsize':(15,8)})
sns.countplot(data=data,x='imdb_score_bucket')
plt.title('imdb_score_bucket vs num_movies')


# <b>
#     <li>From above plot we can observe most of the movies have average ratings</li>
# </b>

# In[95]:


# data.columns
data.groupby(['imdb_score_bucket'])['budget'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='amax',ascending=False).head(5)


# <b>
#     <li>From above data we can see that not heavy budgeted movie will have high rating</li>
#     <li>here movies which have imdb ratings greater than 9 have moderate budgets</li>
# </b>

# In[97]:


# data.columns
data.groupby(['imdb_score_bucket'])['gross'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='amax',ascending=False).head(5)


# In[101]:


# data.columns
data.groupby(['imdb_score_bucket'])['gross'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='mean',ascending=False).head(5)


# <b>
#     <li>From above results we can see that movies which have high imdb ratings has collected high gross</li>
# </b>

# In[109]:


#gross
# data['num_voted_users']
sns.scatterplot(data['gross'],data['num_voted_users'])


# In[111]:


data.groupby('title_year')['num_voted_users'].agg([np.min,np.max,np.mean,np.median]).reset_index().sort_values(by='mean',ascending=False).head(5)


# ### Genres

# In[125]:


#https://www.kaggle.com/code/abhi98/eda-on-imdb-dataset/notebook
from collections import defaultdict
default_dict = defaultdict(list)

tmp = data['genres'].str.split('|')
tmp_dict = tmp.to_dict()

for idx, genres in tmp_dict.items():
    for g in genres:
        default_dict[g].append(idx)


# In[128]:


# default_dict
total_gross_per_genre = pd.DataFrame([[genre, data.loc[default_dict[genre]].gross.sum()] for genre in default_dict.keys()], columns=['genre', 'gross_mean'])
total_gross_per_genre = total_gross_per_genre.dropna().sort_values(by='gross_mean').reset_index(drop=True)
# total_gross_per_genre


# In[131]:


total_gross_per_genre.plot.barh(x='genre', y='gross_mean', figsize=(14, 8), color=['purple'], legend=False)
plt.xlabel('Total Box-Office Collection');


# <b>
#     <li>from above plot we can observe that movies which have adventure genre have collected more at box office</li>
# </b>

# In[136]:


data[['aspect_ratio']].describe()


# In[139]:


# data[['num_critic_for_reviews','num_user_for_reviews']]
#gross
# data['num_voted_users']
sns.scatterplot(data['gross'],data['num_critic_for_reviews'])


# <b>we cant infer any insights from above plot</b>

# In[143]:


len(data['actor_2_name'].unique())

