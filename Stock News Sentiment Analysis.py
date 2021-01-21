#!/usr/bin/env python
# coding: utf-8

# In[23]:


#pip install nltk


# Import Libraries:

# In[1]:


from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
finwiz_url='https://finviz.com/quote.ashx?t='


# Store the Date, Time and News Headlines Data:
# 

# The code below shows stores the entire ‘news-table’ from the FinViz website into a Python dictionary, news_tables, for theses stocks — Amazon (AMZN), Tesla (TSLA) and Google(GOOG)

# In[7]:


news_tables = {}
tickers = ['AMZN','GOOG','TSLA']

for ticker in tickers:
    url = finwiz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    # Add the table to our dictionary
    news_tables[ticker] = news_table


# Print the Data Stored in news_tables:
# 

# To get a sense of what is stored in the news_tables dictionary for ‘AMZN’, which iterates through each <tr></tr> tags (for the first 4 rows) to obtain the headlines between the <a></a> tags and the date and time between the <td></td> tags before printing them out. 

# In[8]:


# Read one single day of headlines for 'AMZN' 
amzn = news_tables['AMZN']
# Get all the table rows tagged in HTML with <tr> into 'amzn_tr'
amzn_tr = amzn.findAll('tr')

for i, table_row in enumerate(amzn_tr):
    # Read the text of the element 'a' into 'link_text'
    a_text = table_row.a.text
    # Read the text of the element 'td' into 'data_text'
    td_text = table_row.td.text
    # Print the contents of 'link_text' and 'data_text' 
    print(a_text)
    print(td_text)
    # Exit after printing 4 rows of data
    if i == 3:
        break


# Parse the Date, Time and News Headlines into a Python List

# It parses the date, time and headlines into a Python list called parsed_news instead of printing it out. The if, else loop is necessary because if you look at the news headlines above, only the first news of each day has the ‘date’ label, the rest of the news only has the ‘time’ label so we have to account for this.

# In[9]:


parsed_news = []

# Iterate through the news
for file_name, news_table in news_tables.items():
    # Iterate through all tr tags in 'news_table'
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() 
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        # Extract the ticker from the file name, get the string up to the 1st '_'  
        ticker = file_name.split('_')[0]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([ticker, date, time, text])
        
parsed_news


# It is actually a list of lists, with each list containing the ticker symbol, date, time and corresponding news-headline.

# Sentiment Analysis with Vader!

# We store the ticker, date, time, headlines in a Pandas DataFrame, perform sentiment analysis on the headlines before adding an additional column in the DataFrame to store the sentiment scores for each headline.

# In[32]:


# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# Set column names
columns = ['ticker', 'date', 'time', 'headline']

# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

# Iterate through the headlines and get the polarity scores using vader
scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

# Join the DataFrames of the news and the list of dicts
parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

# Convert the date column from string to datetime
parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

parsed_and_scored_news.head()


# The ‘compound’ column gives the sentiment scores. For positive scores, the higher the value, the more positive the sentiment is. Similarly for negative scores, the more negative the value, the more negative the sentiment is. The scores range from -1 to 1.

# Plot a Bar Chart of the Sentiment Score for Each Day:

# The average of the sentiment scores for all news headlines collected during each date and plot it on a bar chart. You can average the scores for each week too, to obtain the overall sentiment for a week.

# In[33]:


plt.rcParams['figure.figsize'] = [10, 6]

# Group by date and ticker columns from scored_news and calculate the mean
mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()

# Unstack the column ticker
mean_scores = mean_scores.unstack()

# Get the cross-section of compound in the 'columns' axis
mean_scores = mean_scores.xs('compound', axis="columns").transpose()

# Plot a bar chart with pandas
mean_scores.plot(kind = 'bar')
plt.grid()


# On some days without news headlines for any particular stock, there would be no sentiment score.

# In[ ]:





# In[ ]:




