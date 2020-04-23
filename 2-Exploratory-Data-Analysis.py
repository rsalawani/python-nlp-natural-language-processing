#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# ## Introduction

# After the data cleaning step where we put our data into a few standard formats, the next step is to take a look at the data and see if what we're looking at makes sense. Before applying any fancy algorithms, it's always important to explore the data first.
# 
# When working with numerical data, some of the exploratory data analysis (EDA) techniques we can use include finding the average of the data set, the distribution of the data, the most common values, etc. The idea is the same when working with text data. We are going to find some more obvious patterns with EDA before identifying the hidden patterns with machines learning (ML) techniques. We are going to look at the following for each comedian:
# 
# 1. **Most common words** - find these and create word clouds
# 2. **Size of vocabulary** - look number of unique words and also how quickly someone speaks
# 3. **Amount of profanity** - most common terms

# ## Most Common Words

# ### Analysis

# In[7]:


# Read in the DTM (Document-Term Matrix) from the previously pickled file
import pandas as pd

data = pd.read_pickle('dtm.pkl')
data = data.transpose()  # Original DTM is transposed for better viewing (fit more words and all comedians in the view)
# data.shape   # (7484, 12) --> 7584 unique words x 12 comedians (unique words may/may not be used by every comedian)
# data.head()  # df.head(0) gives Top 5 (default) rows (standard Pandas)
data.head(15)  # Top 15 rows (custom specified)


# In[16]:


# Find the TOP 30 words said by EACH comedian
top_dict = {}  # An empty dictionary that will be populated with comedians (keys) and their TOP 30 words (values)
for cmdn in data.columns:
    top = data[cmdn].sort_values(ascending=False).head(30)
    top_dict[cmdn] = list(zip(top.index, top.values))

top_dict


# In[15]:


# Print the TOP 15 words said by each comedian
for cmdn, top_words in top_dict.items():
    print(cmdn)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')


# **NOTE:** At this point, we could go on and create word clouds. However, by looking at these top words, you can see that some of them have very little meaning and could be added to a stop words list, so let's do just that.
# 
# 

# In[17]:


# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the TOP 30 words for EACH comedian
words = []
for cmdn in data.columns:
    top = [word for (word, count) in top_dict[cmdn]]
    for t in top:
        words.append(t)
        
words


# In[18]:


# Let's AGGREGATE this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()


# In[19]:


# If more than half of the 12 comedians have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
add_stop_words


# In[23]:


# Let's update our Document-Term Matrix (DTM) with the new list of STOP words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Read in previousy cleaned and pickled data
data_clean = pd.read_pickle('data_clean.pkl')

# Add new STOP words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix (remember old parameter stop_words='english')
cv = CountVectorizer(stop_words=stop_words) # Exclude all STOP words the newly updated list of STOP Words 
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")


# In[24]:


# Let's make some WORD CLOUDS!
# PowerShell Prompt: python -m pip install wordcloud "AND" Anaconda Prompt: conda install -c conda-forge wordcloud

rom wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2", max_font_size=150, random_state=42)


# In[25]:


# Reset the output dimensions
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 6]

full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

# Create subplots for each comedian
for index, comedian in enumerate(data.columns):
    wc.generate(data_clean.transcript[comedian])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")  # wc is the WordCloud object generated in earlier step
    plt.axis("off")
    plt.title(full_names[index])
    
plt.show()


# ### Findings

# * Ali Wong says the s-word a lot and talks about her husband. I guess that's funny to me.
# * A lot of people use the F-word. Let's dig into that later.

# ## Number of Words

# ### Analysis

# In[44]:


# Find the Number of UNIQUE WORDS used by EACH comedian

# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for comedian in data.columns:
    uniques = data[comedian].to_numpy().nonzero()[0].size  # Added '.to_numpy()' in the code to fix an AttributeError
    unique_list.append(uniques)

# Create a new dataframe that contains this UNIQUE WORD Count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
data_unique_sort


# In[45]:


# Calculate the Words-per-minute for EACH comedian

# Find the total number of words that a comedian uses
total_list = []
for comedian in data.columns:
    totals = sum(data[comedian])
    total_list.append(totals)
    
# Comedy special run times from IMDB, in minutes
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')
data_wpm_sort


# In[46]:


# Let's plot our findings
import numpy as np

y_pos = np.arange(len(data_words))

plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.comedian)
plt.title('Number of Unique Words', fontsize=20)

plt.subplot(1, 2, 2)
plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
plt.yticks(y_pos, data_wpm_sort.comedian)
plt.title('Number of Words Per Minute', fontsize=20)

plt.tight_layout()
plt.show()


# ### Findings

# * **Vocabulary**
#    * Ricky Gervais (British comedy) and Bill Burr (podcast host) use a lot of words in their comedy
#    * Louis C.K. (self-depricating comedy) and Anthony Jeselnik (dark humor) have a smaller vocabulary
# 
# 
# * **Talking Speed**
#    * Joe Rogan (blue comedy) and Bill Burr (podcast host) talk fast
#    * Bo Burnham (musical comedy) and Anthony Jeselnik (dark humor) talk slow
#    
# Ali Wong is somewhere in the middle in both cases. Nothing too interesting here.

# ## Amount of Profanity

# ### Analysis

# In[47]:


# Earlier I said we'd revisit profanity. Let's take a look at the most common words again.
Counter(words).most_common()


# In[48]:


# Let's isolate just these bad words
data_bad_words = data.transpose()[['fucking', 'fuck', 'shit']]
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, data_bad_words.shit], axis=1)
data_profanity.columns = ['f_word', 's_word']
data_profanity


# In[49]:


# Let's create a scatter plot of our findings
plt.rcParams['figure.figsize'] = [10, 8]

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+1.5, y+0.5, full_names[i], fontsize=10)
    plt.xlim(-5, 155) 
    
plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()


# ### Findings

# * **Averaging 2 F-Bombs Per Minute!** - I don't like too much swearing, especially the f-word, which is probably why I've never heard of Bill Bur, Joe Rogan and Jim Jefferies.
# * **Clean Humor** - It looks like profanity might be a good predictor of the type of comedy I like. Besides Ali Wong, my two other favorite comedians in this group are John Mulaney and Mike Birbiglia.

# ## Side Note

# What was our goal for the EDA portion of our journey? **To be able to take an initial look at our data and see if the results of some basic analysis made sense.**
# 
# My conclusion - yes, it does, for a first pass. There are definitely some things that could be better cleaned up, such as adding more stop words or including bi-grams. But we can save that for another day. The results, especially the profanity findings, are interesting and make general sense, so we're going to move on.
# 
# As a reminder, the data science process is an interative one. It's better to see some non-perfect but acceptable results to help you quickly decide whether your project is a dud or not, instead of having analysis paralysis and never delivering anything.
# 
# **Alice's data science (and life) motto: Let go of perfectionism!**

# ## Additional Exercises

# 1. What other word counts do you think would be interesting to compare instead of the f-word and s-word? Create a scatter plot comparing them.

# In[ ]:




