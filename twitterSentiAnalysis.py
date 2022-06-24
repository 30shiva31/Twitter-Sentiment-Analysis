import pandas as pd # It is used to work on data frames(excel sheets)
import numpy as np # Used for numerical analysis
import seaborn as sns # Used for advance visualization
import matplotlib.pyplot as plt # Used for data visualization
from jupyterthemes import jtplot 
jtplot.style(theme='monokai',context='notebook',ticks=True,grid=False) #setting the theme to momokai makes the graph of x and y visible as we have a dark background, else the x and y axiz would be dark same as the background and it would be hard to see
 
#Loading the data
tweets_df=pd.read_csv('twitter.csv')
print(tweets_df)    #     OUTPUT
print(tweets_df.info())  #summary of information that is in data set
print(tweets_df.describe())  #gives statistical summary of the data such as, count, mean, sd, min, percentile's, max
print(tweets_df['tweet']) #to access an entire column by selecting it
tweets_df=tweets_df.drop(['id'],axis=1) # we are selecting the tweets_df data frame and selecting a column named id, and dropping the entire column using axis=1, axis=0 is used to drop an entire row
print(tweets_df) #checking if the id is dropped


# Performing data exploration
sns.heatmap(tweets_df.isnull(),yticklabels=False,cbar=False,cmap="Blues") #Checking for the prsesnce of ay null elements, in the data set
plt.show() #printing the graph