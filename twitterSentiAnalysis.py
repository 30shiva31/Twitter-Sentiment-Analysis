import pandas as pd # It is used to work on data frames(excel sheets)
import numpy as np # Used for numerical analysis
import seaborn as sns # Used for advance visualization
import matplotlib.pyplot as plt # Used for data visualization
from jupyterthemes import jtplot 
jtplot.style(theme='monokai',context='notebook',ticks=True,grid=False) #setting the theme to momokai makes the graph of x and y visible as we have a dark background, else the x and y axiz would be dark same as the background and it would be hard to see
 
#Loading the data
tweets_df=pd.read_csv('twitter.csv')
#print(tweets_df)         OUTPUT