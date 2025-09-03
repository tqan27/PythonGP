import os
import pandas as pd
from collections import Counter
from data_loader import load_data
from sentiment_analyzer import analyze_sentiment, get_sentiment_distribution, extract_keywords
from visualizer import (
    plot_sentiment_distribution, 
    plot_sentiment_trend,        
    plot_sentiment_bar,           # New added 
    plot_top_keywords             # New added 
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    input_path = 'sentiment_report.csv'
    output_dir = './output'
    ensure_dir(output_dir)

    # 读取数据
    df = load_data(input_path)
    if df is None:
        return

    # 检查 Review 列
    if 'Review' not in df.columns:
        raise ValueError("Expected column 'Review' not found in dataset")
    df = df.dropna(subset=['Review']).copy()

    # edit过这里。
    df['sentiment'], df['polarity'] = zip(*df['Review'].apply(analyze_sentiment))

    # edit 过这里
    processed_path = os.path.join(output_dir, 'sentiment_report.csv')
    df.to_csv(processed_path, index=False)

    # Sentiment distribution (predicted)
    sentiment_counts = get_sentiment_distribution(df)
    print("Sentiment Distribution:", sentiment_counts)

    # ===== Plot Sentiment Distribution (Pie Chart) =====
    plot_sentiment_distribution(
        sentiment_counts, 
        os.path.join(output_dir, 'sentiment_distribution.png')
    )

    # ===== (New added) =====
    plot_sentiment_bar(
        sentiment_counts, 
        os.path.join(output_dir, 'sentiment_bar.png')
    )


    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df.index, unit='D')

    plot_sentiment_trend(
        df.copy(), 
        os.path.join(output_dir, 'sentiment_trend.png')
    )

    all_keywords: list[str] = []
    for review in df['Review']:
        all_keywords.extend(extract_keywords(review))
    key_themes = Counter(all_keywords).most_common(20)  # Top 20 themes
    print("Top Key Themes:", key_themes[:10])

    # (New added) 
    plot_top_keywords(
        key_themes, 
        os.path.join(output_dir, 'top_keywords.png')
    )

    metrics_lines = []
    if 'label' in df.columns:
        label_map = {'POSITIVE': 'POSITIVE', 'NEGATIVE': 'NEGATIVE', 'NEUTRAL': 'NEUTRAL'}
        eval_df = df[df['label'].isin(label_map.keys())].copy()
        eval_df['label_norm'] = eval_df['label'].map(label_map)
        accuracy = (eval_df['label_norm'] == eval_df['sentiment']).mean()
        from sklearn.metrics import classification_report, confusion_matrix
        try:
            report = classification_report(eval_df['label_norm'], eval_df['sentiment'], zero_division=0)
            cm = confusion_matrix(
                eval_df['label_norm'], 
                eval_df['sentiment'], 
                labels=['NEGATIVE', 'NEUTRAL', 'POSITIVE']
            )
            metrics_lines.append(f"Accuracy: {accuracy:.4f}")
            metrics_lines.append("\nClassification Report:\n" + report)
            metrics_lines.append(
                "Confusion Matrix (rows=true, cols=pred) [NEGATIVE, NEUTRAL, POSITIVE]:\n" + str(cm)
            )
        except Exception as e:
            metrics_lines.append(f"Accuracy: {accuracy:.4f} (detailed metrics unavailable: {e})")
    else:
        metrics_lines.append("Ground-truth labels not found; skipped evaluation metrics.")

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Sentiment counts (predicted):\n")
        for k, v in sentiment_counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTop key themes:\n")
        for word, cnt in key_themes:
            f.write(f"  {word}: {cnt}\n")
        f.write("\n" + "\n".join(metrics_lines) + "\n")

    print(f"Processed data saved to: {processed_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Figures saved to: {output_dir}")

if __name__ == '__main__':
    main()