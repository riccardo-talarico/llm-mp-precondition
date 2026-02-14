import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def print_results(file_name_results='benchmark_results_qwen/qwen3-32b_clean.csv'):
    df_true = pd.read_csv("./benchmark_classification.csv")
    df_results = pd.read_csv(file_name_results)

    y_true = df_true['subtype'].str.strip()
    y_pred = df_results['subtype'].str.strip().fillna('None')

    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, zero_division=0)

    missing_vals = df_results[['subtype','subsubtype']].isnull().sum()
    print(f"Missing values for qwen3: {missing_vals}/68")

    print(f"Weighted F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)




if __name__ == '__main__':
    print_results()
