import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report


def print_results(file_name_results='benchmark_results_qwen/qwen3-32b_clean.csv'):
    df_true = pd.read_csv("./benchmark_classification.csv")
    df_results = pd.read_csv(file_name_results)

    y_true = df_true['subtype'].str.strip()
    y_pred = df_results['subtype'].str.strip().fillna('None')

    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    missing_vals = df_results[['subtype','subsubtype']].isnull().sum()
    print(f"Missing values for {file_name_results}: {missing_vals}/68")

    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy Score: {acc:.4f}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == '__main__':
    print_results()
    print_results('benchmark_results_llama-3.3-70b-versatile_clean.csv')
