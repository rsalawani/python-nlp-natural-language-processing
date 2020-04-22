#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# ## Introduction

# This notebook goes through a necessary step of any data science project - data cleaning. Data cleaning is a time consuming and unenjoyable task, yet it's a very important one. Keep in mind, "garbage in, garbage out". Feeding dirty data into a model will give us results that are meaningless.
# 
# Specifically, we'll be walking through:
# 
# 1. **Getting the data - **in this case, we'll be scraping data from a website
# 2. **Cleaning the data - **we will walk through popular text pre-processing techniques
# 3. **Organizing the data - **we will organize the cleaned data into a way that is easy to input into other algorithms
# 
# The output of this notebook will be clean, organized data in two standard text formats:
# 
# 1. **Corpus** - a collection of text
# 2. **Document-Term Matrix** - word counts in matrix format

# ## Problem Statement

# As a reminder, our goal is to look at transcripts of various comedians and note their similarities and differences. Specifically, I'd like to know if Ali Wong's comedy style is different than other comedians, since she's the comedian that got me interested in stand up comedy.

# ## Getting The Data

# Luckily, there are wonderful people online that keep track of stand up routine transcripts. [Scraps From The Loft](http://scrapsfromtheloft.com) makes them available for non-profit and educational purposes.
# 
# To decide which comedians to look into, I went on IMDB and looked specifically at comedy specials that were released in the past 5 years. To narrow it down further, I looked only at those with greater than a 7.5/10 rating and more than 2000 votes. If a comedian had multiple specials that fit those requirements, I would pick the most highly rated one. I ended up with a dozen comedy specials.

# In[6]:


# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
import pickle

# Scrapes transcript data from the website 'scrapsfromtheloft.com'
def url_to_transcript(url):
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [prgf.text for prgf in soup.find(class_="post-content").find_all('p')]  # Collect ALL paragraphs from HTML 'p' div-tag (section) 'post-content' of the webpage
    print(url)
    return text

# [List] of URLs of transcripts in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

# Comedian names [listed] in the same order as above URLs
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']


# In[ ]:


# Actually request transcripts (takes a few minutes to run)
# NOTE: These transcripts have been created and saved on the hard disk already.
# Uncomment the below code if you want to re-create them

# Create a [list] of transcripts using List Comprehension
transcripts = [url_to_transcript(url) for url in urls]


# In[ ]:


# Pickle (save) files for later use
# NOTE: These transcripts have been created and saved (pickled) on the hard disk already.
# Uncomment the below code if you want to re-create them

# Make a new directory to hold the text files
get_ipython().system('mkdir transcripts')

for i, cmdn in enumerate(comedians):
    with open("transcripts/" + cmdn + ".txt", "wb") as file:
        pickle.dump(transcripts[i], file)


# In[8]:


# Load pickled files and create a dictionary of the transcript files where KEY = comedian, and VALUE = transcript text
data = {}
for i, cmdn in enumerate(comedians):
    with open("transcripts/" + cmdn + ".txt", "rb") as file:
        data[cmdn] = pickle.load(file)


# In[9]:


# Double check to make sure data has been loaded properly
data.keys()


# In[10]:


# More checks
data['louis'][:2]


# ## Cleaning The Data

# When dealing with numerical data, data cleaning often involves removing null values and duplicate data, dealing with outliers, etc. With text data, there are some common data cleaning techniques, which are also known as text pre-processing techniques.
# 
# With text data, this cleaning process can go on forever. There's always an exception to every cleaning step. So, we're going to follow the MVP (minimum viable product) approach - start simple and iterate. Here are a bunch of things you can do to clean your data. We're going to execute just the common cleaning steps here and the rest can be done at a later point to improve our results.
# 
# **Common data cleaning steps on all text:**
# * Make text all lower case
# * Remove punctuation
# * Remove numerical values
# * Remove common non-sensical text (/n)
# * Tokenize text
# * Remove stop words
# 
# **More data cleaning steps after tokenization:**
# * Stemming / lemmatization
# * Parts of speech tagging
# * Create bi-grams or tri-grams
# * Deal with typos
# * And more...

# In[11]:


# Let's take a look at our data again
next(iter(data.keys()))


# In[12]:


# Notice that our dictionary is currently in key: comedian, value: list of text format
next(iter(data.values()))


# In[13]:


# We are going to change this to key: comedian, value: string format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text


# In[14]:


# Combine it!
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}


# In[33]:


# We can either keep it in dictionary format or put it into a pandas dataframe
import pandas as pd
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df  # Displays the df where the actual transcript is just in the 'corpus' (continuous text string) form


# In[16]:


# Let's take a look at the transcript for Ali Wong
data_df.transcript.loc['ali']


# In[17]:


# Apply a first round of text cleaning techniques
import re  # Import Regular Expressions so we can identify text patterns that we can ignore/delete 
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()  # Convert all text to lower case 
    text = re.sub('\[.*?\]', '', text)  # Replace (delete) all annotations within []
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Replace (delete) all punctuation marks
    text = re.sub('\w*\d\w*', '', text)  # Replace (delete) all numbers and alpha-numeric combinations
    return text  # Return what's left after deleting all the unnecessary characters/text

round1 = lambda x: clean_text_round1(x)


# In[18]:


# Let's take a look at the updated text
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
data_clean


# In[30]:


# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)  # Replace (delete) all single and double quotes and elipses
    text = re.sub('\n', '', text)  # Replace (delete) all newline characters
    return text  # Return what's left after deleting all the unnecessary characters

round2 = lambda x: clean_text_round2(x)


# In[31]:


# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
data_clean


# **NOTE:** This data cleaning aka text pre-processing step could go on for a while, but we are going to stop for now. After going through some analysis techniques, if you see that the results don't make sense or could be improved, you can come back and make more edits such as:
# * Mark 'cheering' and 'cheer' as the same word (stemming / lemmatization)
# * Combine 'thank you' into one term (bi-grams)
# * And a lot more...

# ## Organizing The Data

# I mentioned earlier that the output of this notebook will be clean, organized data in two standard text formats:
# 1. **Corpus - **a collection of text
# 2. **Document-Term Matrix - **word counts in matrix format

# ### Corpus

# We already created a corpus in an earlier step. The definition of a corpus is a collection of texts, and they are all put together neatly in a pandas dataframe here.

# In[35]:


# Let's take a look at our (original uncleaned) dataframe
data_df


# In[22]:


# Let's add the comedians' full names as well
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

data_df['full_name'] = full_names
data_df  # Display uncleaned DF with added column for comedian's full name


# In[23]:


# Let's pickle it for later use
data_df.to_pickle("corpus.pkl")


# ### Document-Term Matrix

# For many of the techniques we'll be using in future notebooks, the text must be tokenized, meaning broken down into smaller pieces. The most common tokenization technique is to break down text into words. We can do this using scikit-learn's CountVectorizer, where every row will represent a different document and every column will represent a different word.
# 
# Also we can remove STOP words using CountVectorizer. STOP words are common filler words ( such as 'a', 'the', pronouns, prepositions, conjuctions, etc.) that may be grammatically necessary in spoken/written context BUT ADD NO VALUE / additional meaning to text.  You can also use CountVectorizer to make separate columns for single words (mono/unigrams), and (hyphenated) bi-grams, tri-grams, etc.
# 
# Click on CountVectorizer withing the script below and simultaneously press Shift + Tab to get more detailed tooltips! 

# In[24]:


# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')  # CountVectorizer ignores ALL English STOP/filler words and punctuations
data_cv = cv.fit_transform(data_clean.transcript)  # Fits the 'cleaned DF' into the CountVectorizer object
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())  # Creates Document-Term Matrix shown below
data_dtm.index = data_clean.index
data_dtm  # Display Document-Term Matrix


# In[27]:


# Let's pickle the cleaned data matrix for later use
data_dtm.to_pickle("dtm.pkl")


# In[28]:


# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))


# ## Additional Exercises

# 1. Can you add an additional regular expression to the clean_text_round2 function to further clean the text?
# 2. Play around with CountVectorizer's parameters. What is ngram_range? What is min_df and max_df?

# In[ ]:




