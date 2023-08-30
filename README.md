# British Airways Reviews Analysis

## Overview
This repository contains Jupyter Notebook files and data for scraping, analyzing, and visualizing customer reviews of British Airways. The analysis includes sentiment analysis, word frequency analysis, and visualization of both positive and negative reviews.

## Contents
- [Web Scraping and Data Collection](#web-scraping-and-data-collection)
- [Data Cleaning](#data-cleaning)
- [Sentiment Analysis](#sentiment-analysis)
- [Word Frequency Analysis](#word-frequency-analysis)
- [WordCloud Visualizations](#wordcloud-visualizations)

## Web Scraping and Data Collection
The Jupyter Notebook [`web_scraping.ipynb`](web_scraping.ipynb) demonstrates how to scrape customer reviews of British Airways from [airlinequality.com](https://www.airlinequality.com/airline-reviews/british-airways). It uses the `BeautifulSoup` library to collect review text and saves the data to a CSV file.

## Data Cleaning
The data collected from web scraping may contain unnecessary text or symbols. The [`data_cleaning.ipynb`](data_cleaning.ipynb) notebook showcases how to clean the review text by removing irrelevant information and symbols.

## Sentiment Analysis
In the [`sentiment_analysis.ipynb`](sentiment_analysis.ipynb) notebook, sentiment analysis is performed on the cleaned reviews using the `TextBlob` library. The sentiment scores (polarity and subjectivity) are calculated and used to categorize reviews as positive, negative, or neutral.

## Word Frequency Analysis
The [`word_frequency_analysis.ipynb`](word_frequency_analysis.ipynb) notebook covers word frequency analysis on both positive and negative reviews. It uses the `Counter` class to count the occurrences of words, filter out stopwords, and visualize the most common words.

## WordCloud Visualizations
The [`wordcloud_visualizations.ipynb`](wordcloud_visualizations.ipynb) notebook demonstrates the creation of WordCloud visualizations for both positive and negative reviews using the `WordCloud` library. WordClouds provide a graphical representation of word frequency.

## Usage
1. Clone this repository to your local machine using `git clone https://github.com/yourusername/british-airways-reviews.git`.
2. Install the required libraries by running `pip install -r requirements.txt`.
3. Follow the notebooks in the order mentioned above to understand the process of data collection, cleaning, analysis, and visualization.
4. Customize the notebooks according to your needs and explore further analyses if desired.

## Credits
- Web scraping: [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- Sentiment analysis: [TextBlob](https://textblob.readthedocs.io/en/dev/)
- WordCloud visualizations: [WordCloud](https://github.com/amueller/word_cloud)
- Data source: [AirlineQuality](https://www.airlinequality.com/airline-reviews/british-airways)

## License
This project is licensed under the [MIT License](LICENSE).
