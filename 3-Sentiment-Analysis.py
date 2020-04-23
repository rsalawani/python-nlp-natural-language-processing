#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis

# ## Introduction

# So far, all of the analysis we've done has been pretty generic - looking at counts, creating scatter plots, etc. These techniques could be applied to numeric data as well.
# 
# When it comes to text data, there are a few popular techniques that we'll be going through in the next few notebooks, starting with sentiment analysis. A few key points to remember with sentiment analysis.
# 
# 1. **TextBlob Module:** Linguistic researchers have labeled the sentiment of words based on their domain expertise. Sentiment of words can vary based on where it is in a sentence. The TextBlob module allows us to take advantage of these labels.
# 2. **Sentiment Labels:** Each word in a corpus is labeled in terms of polarity and subjectivity (there are more labels as well, but we're going to ignore them for now). A corpus' sentiment is the average of these.
#    * **Polarity**: How positive or negative a word is. -1 is very negative. +1 is very positive.
#    * **Subjectivity**: How subjective, or opinionated a word is. 0 is fact. +1 is very much an opinion.
# 
# For more info on how TextBlob coded up its [sentiment function](https://planspace.org/20150607-textblob_sentiment/).
# 
# Let's take a look at the sentiment of the various transcripts, both overall and throughout the comedy routine.

# ## Sentiment of Routine

# In[1]:


# We'll start by reading in the corpus, which preserves word order
import pandas as pd

data = pd.read_pickle('corpus.pkl')
data


# In[2]:


# Create quick lambda functions to find the polarity and subjectivity of each routine
# Anaconda Promot: conda install -c conda-forge textblob  AND  PowerShell Prompt: python -m pip install textblob

from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)
data


# In[3]:


# Let's plot the results
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

for index, comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


# ## Sentiment of Routine Over Time

# Instead of looking at the overall sentiment, let's see if there's anything interesting about the sentiment over time throughout each routine.

# In[4]:


# Split each routine into 10 parts
import numpy as np
import math

def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list


# In[5]:


# Let's take a look at our data again
data


# In[6]:


# Let's create a list to hold all of the pieces of text
list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)
    
list_pieces


# In[7]:


# The list has 10 elements, one for each transcript
len(list_pieces)


# In[8]:


# Each transcript has been split into 10 pieces of text
len(list_pieces[0])


# In[9]:


# Calculate the polarity for each piece of text

polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)
    
polarity_transcript


# In[10]:


# Show the plot for one comedian
plt.plot(polarity_transcript[0])
plt.title(data['full_name'].index[0])
plt.show()


# In[11]:


# Show the plot for all comedians
plt.rcParams['figure.figsize'] = [16, 12]

for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)
    
plt.show()


# Ali Wong stays generally positive throughout her routine. Similar comedians are Louis C.K. and Mike Birbiglia.
# 
# On the other hand, you have some pretty different patterns here like Bo Burnham who gets happier as time passes and Dave Chappelle who has some pretty down moments in his routine.

# ## Additional Exercises

# 1. Modify the number of sections the comedy routine is split into and see how the charts over time change.

# In[ ]:




