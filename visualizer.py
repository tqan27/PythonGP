import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sentiment_distribution(sentiment_counts, output_path='sentiment_distribution.png'):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#66b3ff', '#ff9999', '#99ff99'] # Blue, Red, Green for Positive, Negative, Neutral

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(output_path)
    plt.close()
    print(f'Sentiment distribution plot saved to {output_path}')

def plot_sentiment_trend(df, output_path='sentiment_trend.png'):
    # Ensure 'timestamp' is datetime and set as index for resampling
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample by day and calculate mean polarity
    daily_sentiment = df['polarity'].resample('D').mean().fillna(0) # Fill NaN with 0 for days with no posts

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=daily_sentiment.index, y=daily_sentiment.values)
    plt.title('Daily Sentiment Trend')
    plt.xlabel('Date')
    plt.ylabel('Average Polarity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Sentiment trend plot saved to {output_path}')

#New added
def plot_sentiment_bar(sentiment_counts, output_path='sentiment_bar.png'):
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    colors = ['#66b3ff', '#ff9999', '#99ff99']  

    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=values, palette=colors)
    plt.title('Sentiment Count Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Sentiment bar plot saved to {output_path}')

#New added
def plot_top_keywords(key_themes, output_path='top_keywords.png', top_n=20):
    words = [w for w, _ in key_themes[:top_n]]
    counts = [c for _, c in key_themes[:top_n]]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette='viridis')
    plt.title(f'Top {top_n} Keywords')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Top keywords plot saved to {output_path}')