from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords  # natural language tool kit
import nltk
import string
from wordcloud import WordCloud  # importing wordcloud
import pandas as pd  # It is used to work on data frames(excel sheets)
import numpy as np  # Used for numerical analysis
import seaborn as sns  # Used for advance visualization
import matplotlib.pyplot as plt  # Used for data visualization
from jupyterthemes import jtplot
# setting the theme to momokai makes the graph of x and y visible as we have a dark background, else the x and y axiz would be dark same as the background and it would be hard to see
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

# Loading the data
tweets_df = pd.read_csv('twitter.csv')
print(tweets_df)  # OUTPUT
print(tweets_df.info())  # summary of information that is in data set
# gives statistical summary of the data such as, count, mean, sd, min, percentile's, max
print(tweets_df.describe())
print(tweets_df['tweet'])  # to access an entire column by selecting it
# we are selecting the tweets_df data frame and selecting a column named id, and dropping the entire column using axis=1, axis=0 is used to drop an entire row
tweets_df = tweets_df.drop(['id'], axis=1)
print(tweets_df)  # checking if the id is dropped

# Performing data exploration

# Checking for the prsesnce of ay null elements, in the data set
sns.heatmap(tweets_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()  # printing the graph
# print(tweets_df.isnull().sum()) # Is used to tell the count of null in the data frame, upper command shows the graph though
# using histogram from matplotlib it shows
tweets_df.hist(bins=30, figsize=(13, 5), color='r')
plt.show()
# tweets_df['label'].hist() # This plots the graph on the column label, same us the above command
# plt.show()
# counts and displays the count in graph
sns.countplot(tweets_df['label'], label='Count')
plt.show()
# finds the length of every tweet and inserts a new column named length beside the tweets respectively
tweets_df['length'] = tweets_df['tweet'].apply(len)
print(tweets_df)
# In this you can see the summary, and you can see the min, max and avg length of the tweets in the data frame
print(tweets_df.describe())
# # viewing the tweet of minimum size(size=11 as shown in describe)
print(tweets_df[tweets_df['length'] == 11]['tweet'])

# Plotting the word cloud

# all the data which has label=0 are kept under a new data frame called positive
positive = tweets_df[tweets_df['label'] == 0]
print(positive)
negative = tweets_df[tweets_df['label'] == 1]
print(negative)
sentences = tweets_df['tweet'].tolist()  # putting all the tweets into a list
print(len(sentences))  # tells us no.of tweets that been into list
# joining all the elements of the list into a single string with a space in b/w
sentences_as_one_string = " ".join(sentences)
print(sentences_as_one_string)
plt.figure(figsize=(20, 20))
# Generating image of the wprd cloud using sentences_as_one_string (All sentences)
plt.imshow(WordCloud().generate(sentences_as_one_string))
plt.show()

# Word cloud using only negative tweets
sentences1 = negative['tweet'].tolist()
sentences_as_one_string1 = " ".join(sentences1)
plt.figure(figsize=(15, 15))
plt.imshow(WordCloud().generate(sentences_as_one_string1))
plt.show()

# Performing data cleaning

# Removing all the punctuations

# printing all the punctuations available in the grammar
print(string.punctuation)
Test = '$I love AI and machine learning!!'
# list comprehension
Test_punc_removed = [char for char in Test if char not in string.punctuation]
# joining the elements of the list
Test_punc_removed_join = ''.join(Test_punc_removed)
print(Test_punc_removed_join)

# Removing all the stop words(these are the common words, that doesnt add much value for analysis)
nltk.download('stopwords')
print(stopwords.words('english'))

Test_punc_removed_join = 'I enjoy coding, programming and Artificial intelligence'
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split(
) if word.lower() not in stopwords.words('english')]  # list comprehension
print(Test_punc_removed_join_clean)  # only important words are left


# Performing count vectorization(Tokenization)

sample_data = ['This is the first paper.', 'This document is the second paper.',
               'And this is the third one.', 'Is this the first paper?']
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(sample_data)
print(vectorizer.get_feature_names())
print(x.toarray())

# practise opp
mini_challenge = ['Hello World',
                  'Hello Hello World', 'Hello World world world']
vectorizer_challenge = CountVectorizer()
x_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(x_challenge.toarray())

# Creating a pipeline to remove punctuations, stopwords and perform count vectorization


def message_cleaning(message):
    test_punc_removed = [
        char for char in message if char not in string.punctuation]
    test_punc_removed_join = ''.join(test_punc_removed)
    test_punc_removed_join_clean = [word for word in test_punc_removed_join.split(
    ) if word.lower() not in stopwords.words('english')]
    return test_punc_removed_join_clean


tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)
print(tweets_df['tweet'][5])  # uncleaned tweet
print(tweets_df_clean[5])  # cleaned data

vectorizer = CountVectorizer(analyzer=message_cleaning, dtype=np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])
print(vectorizer.get_feature_names())
print(tweets_countvectorizer.toarray())
print(tweets_countvectorizer.shape)
x = pd.DataFrame(tweets_countvectorizer.toarray())
print(x)

# Trainand evaluate a naive bayes classifier model

print(x.shape)
y = tweets_df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
NB_classifier = MultinomialNB()
NB_classifier.fit(x_train, y_train)
y_predict_test = NB_classifier.predict(x_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(y_test, y_predict_test))
