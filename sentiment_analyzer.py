import pandas as pd
from textblob import TextBlob
from collections import Counter
import re

def analyze_sentiment(text):
    if pd.isna(text):
        return None, None
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'POSITIVE', polarity
    elif polarity < 0:
        return 'NEGATIVE', polarity
    else:
        return 'NEUTRAL', polarity

def get_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    return sentiment_counts

def extract_keywords(text, num_keywords=5):
    if pd.isna(text):
        return []
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    words = text.split()
    # Filter out common stopwords (can be expanded)
    stopwords = set(['the', 'a', 'an', 'is', 'it', 'in', 'of', 'for', 'and', 'to', 'with', 'on', 'this', 'that', 'be', 'are', 'was', 'were', 'as', 'but', 'not', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose', 'from', 'at', 'by', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    # Count word frequency
    word_counts = Counter(filtered_words)
    # Return most common words
    return [word for word, count in word_counts.most_common(num_keywords)]


