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
