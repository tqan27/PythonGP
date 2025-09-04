import os
import pandas as pd
from collections import Counter

# --- Safe imports ---
try:
    from data_loader import load_data
except ModuleNotFoundError:
    print("[FATAL] Missing required file: data_loader.py. Please ensure it exists in the same directory.")
    exit(1)
except ImportError as e:
    print(f"[FATAL] Could not import load_data from data_loader: {e}")
    exit(1)

try:
    from sentiment_analyzer import analyze_sentiment, get_sentiment_distribution, extract_keywords
except ModuleNotFoundError:
    print("[FATAL] Missing required file: sentiment_analyzer.py. Please ensure it exists.")
    exit(1)
except ImportError as e:
    print(f"[FATAL] Could not import functions from sentiment_analyzer: {e}")
    exit(1)

try:
    from visualizer import (
        plot_sentiment_distribution, 
        plot_sentiment_trend,        
        plot_sentiment_bar,           
        plot_top_keywords             
    )
except ModuleNotFoundError:
    print("[FATAL] Missing required file: visualizer.py. Please ensure it exists.")
    exit(1)
except ImportError as e:
    print(f"[FATAL] Could not import functions from visualizer: {e}")
    exit(1)


def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Failed to create directory '{path}': {e}")
        raise


def main():
    input_path = 'sentiment_report.csv'
    output_dir = './output'

    try:
        ensure_dir(output_dir)
    except Exception:
        print("[FATAL] Could not prepare output directory. Exiting.")
        return

    # Step 1: Load data
    try:
        df = load_data(input_path)
        if df is None or df.empty:
            print(f"[ERROR] No data loaded from {input_path}. Exiting.")
            return
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_path}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error while loading data: {e}")
        return

    # Step 2: Validate columns
    if 'Review' not in df.columns:
        print("[ERROR] Expected column 'Review' not found in dataset. Exiting.")
        return

    try:
        df = df.dropna(subset=['Review']).copy()
    except Exception as e:
        print(f"[ERROR] Failed to clean Review column: {e}")
        return

    # Step 3: Sentiment analysis
    try:
        df['sentiment'], df['polarity'] = zip(*df['Review'].apply(analyze_sentiment))
    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}")
        return

    # Step 4: Save processed data
    processed_path = os.path.join(output_dir, 'sentiment_report.csv')
    try:
        df.to_csv(processed_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to save processed CSV: {e}")
        return

    # Step 5: Sentiment distribution
    try:
        sentiment_counts = get_sentiment_distribution(df)
        print("Sentiment Distribution:", sentiment_counts)

        plot_sentiment_distribution(
            sentiment_counts,
            os.path.join(output_dir, 'sentiment_distribution.png')
        )

        plot_sentiment_bar(
            sentiment_counts,
            os.path.join(output_dir, 'sentiment_bar.png')
        )
    except Exception as e:
        print(f"[WARNING] Failed to generate sentiment distribution plots: {e}")

    # Step 6: Sentiment trend
    try:
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df.index, unit='D')

        plot_sentiment_trend(
            df.copy(),
            os.path.join(output_dir, 'sentiment_trend.png')
        )
    except Exception as e:
        print(f"[WARNING] Failed to generate sentiment trend plot: {e}")

    # Step 7: Keyword extraction
    try:
        all_keywords: list[str] = []
        for review in df['Review']:
            try:
                all_keywords.extend(extract_keywords(review))
            except Exception as sub_e:
                print(f"[WARNING] Keyword extraction failed for review: {sub_e}")

        key_themes = Counter(all_keywords).most_common(20)
        print("Top Key Themes:", key_themes[:10])

        plot_top_keywords(
            key_themes,
            os.path.join(output_dir, 'top_keywords.png')
        )
    except Exception as e:
        print(f"[WARNING] Failed to extract or plot keywords: {e}")
        key_themes = []

    # Step 8: Metrics evaluation
    metrics_lines = []
    try:
        if 'label' in df.columns:
            from sklearn.metrics import classification_report, confusion_matrix
            label_map = {'POSITIVE': 'POSITIVE', 'NEGATIVE': 'NEGATIVE', 'NEUTRAL': 'NEUTRAL'}
            eval_df = df[df['label'].isin(label_map.keys())].copy()
            eval_df['label_norm'] = eval_df['label'].map(label_map)

            accuracy = (eval_df['label_norm'] == eval_df['sentiment']).mean()
            try:
                report = classification_report(
                    eval_df['label_norm'], eval_df['sentiment'], zero_division=0
                )
                cm = confusion_matrix(
                    eval_df['label_norm'],
                    eval_df['sentiment'],
                    labels=['NEGATIVE', 'NEUTRAL', 'POSITIVE']
                )
                metrics_lines.append(f"Accuracy: {accuracy:.4f}")
                metrics_lines.append("\nClassification Report:\n" + report)
                metrics_lines.append(
                    "Confusion Matrix (rows=true, cols=pred) "
                    "[NEGATIVE, NEUTRAL, POSITIVE]:\n" + str(cm)
                )
            except Exception as eval_e:
                metrics_lines.append(
                    f"Accuracy: {accuracy:.4f} (detailed metrics unavailable: {eval_e})"
                )
        else:
            metrics_lines.append("Ground-truth labels not found; skipped evaluation metrics.")
    except Exception as e:
        metrics_lines.append(f"[ERROR] Metrics evaluation failed: {e}")

    # Step 9: Save summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Sentiment counts (predicted):\n")
            for k, v in sentiment_counts.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nTop key themes:\n")
            for word, cnt in key_themes:
                f.write(f"  {word}: {cnt}\n")
            f.write("\n" + "\n".join(metrics_lines) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to save summary file: {e}")
        return

    # Final logs
    print(f"[INFO] Processed data saved to: {processed_path}")
    print(f"[INFO] Summary saved to: {summary_path}")
    print(f"[INFO] Figures saved to: {output_dir}")


if __name__ == '_main_':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")