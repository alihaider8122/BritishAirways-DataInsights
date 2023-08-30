#!/usr/bin/env python
# coding: utf-8

# # Task 1
# 
# ---
# 
# ## Web scraping and analysis
# 
# This Jupyter notebook includes some code to get you started with web scraping. We will use a package called `BeautifulSoup` to collect the data from the web. Once you've collected your data and saved it into a local `.csv` file you should start with your analysis.
# 
# ### Scraping data from Skytrax
# 
# If you visit [https://www.airlinequality.com] you can see that there is a lot of data there. For this task, we are only interested in reviews related to British Airways and the Airline itself.
# 
# If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, we can use `Python` and `BeautifulSoup` to collect all the links to the reviews and then to collect the text data on each of the individual review links.

# In[3]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[ ]:





# In[ ]:


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 11
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")


# In[ ]:


df = pd.DataFrame()
df["reviews"] = reviews
df.head()


# In[ ]:


import os

# Create the 'data' directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
df.to_csv("data/BA_reviews.csv")


# Congratulations! Now you have your dataset for this task! The loops above collected 1000 reviews by iterating through the paginated pages on the website. However, if you want to collect more data, try increasing the number of pages!
# 
#  The next thing that you should do is clean this data to remove any unnecessary text from each of the rows. For example, "✅ Trip Verified" can be removed from each row if it exists, as it's not relevant to what we want to investigate.

# In[ ]:


print(df)


# In[ ]:


for index, row in df.iterrows():
    # Remove "✅ Trip Verified" from the review text
    cleaned_review = row["reviews"].replace("✅ Trip Verified", "").strip()
    cleaned_review = row["reviews"].replace("Not Verified", "").strip()
    # Update the DataFrame with the cleaned review
    df.at[index, "reviews"] = cleaned_review

# Save the cleaned DataFrame to a new CSV file
df.to_csv("data/BA_reviews_cleaned.csv", index=False)


# In[ ]:


print(df)


# In[ ]:


pd.set_option('display.max_colwidth', None)
print(df.iloc[1])


# In[ ]:


import pandas as pd
from textblob import TextBlob


# In[ ]:


# Load dataset into a pandas DataFrame
df = pd.read_csv("data/BA_reviews_cleaned.csv")  # Replace with your dataset filename

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Apply sentiment analysis and create new columns
df["Sentiment_Polarity"], df["Sentiment_Subjectivity"] = zip(*df["reviews"].apply(analyze_sentiment))


# In[ ]:


df["Sentiment_Label"] = df["Sentiment_Polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))


# In[ ]:


print(df)


# In[ ]:


positive=df[df['Sentiment_Label' ]=="Positive"]


# In[ ]:


negative=df[df['Sentiment_Label' ]=="Negative"]


# In[ ]:


print(positive)


# In[ ]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# In[ ]:


# Combine positive reviews into a single text
positive_reviews_text = " ".join(positive["reviews"])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_reviews_text)

# Display the WordCloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


# Combine positive reviews into a single text
positive_reviews_text = " ".join(positive["reviews"])

# Tokenize the text and remove stopwords
words = positive_reviews_text.split()
filtered_words = [word for word in words if word.lower() not in STOPWORDS and len(word) > 2]

# Calculate word frequencies
word_freq = {}
for word in filtered_words:
    if word not in word_freq:
        word_freq[word] = 1
    else:
        word_freq[word] += 1

# Create lists for x and y values
x_values = []
y_values = []
marker_sizes = []

# Populate x and y lists
for word, freq in word_freq.items():
    x_values.append(freq)  # Frequency of word
    y_values.append(positive_sentiment_df["Sentiment_Polarity"].mean())  # Average Sentiment_Polarity
    marker_sizes.append(freq)  # Use frequency as marker size

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, s=marker_sizes, alpha=0.5)
plt.xlabel("Word Frequency")
plt.ylabel("Sentiment Polarity")
plt.title("Scatter Plot: Word Frequency vs Sentiment Polarity")
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:


from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  


# In[ ]:


plt.show()


# In[ ]:




