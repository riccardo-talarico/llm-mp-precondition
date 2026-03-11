import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

def hierarchical_accuracy(df):
    score = 0.0

    for _, row in df.iterrows():
        if row["subtype_true"] == row["subtype_pred"]:
            if row["subsubtype_true"] == row["subsubtype_pred"]:
                score += 1.0
            else:
                score += 0.5

    return score / len(df)


def print_results(file_name_results='benchmark_results_qwen/qwen3-32b_clean.csv'):
    df_true = pd.read_csv("./benchmark_classification.csv")
    df_results = pd.read_csv(file_name_results)

    df = (
    df_true
        .merge(df_results, on="id", how="inner", suffixes=("_true", "_pred"))
    )


    y_true = df_true['subtype'].str.strip()
    y_pred = df_results['subtype'].str.strip().fillna('None')

    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    hacc = hierarchical_accuracy(df)

    missing_vals = df_results[['subtype']].isnull().sum()
    print(f"Missing values for {file_name_results}: {int(missing_vals.iloc[0])}/68 = {float(missing_vals.iloc[0])*100/68:.4f}%")

    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy Score: {acc:.4f}")
    print(f"Hierarchical Accuracy Score: {hacc:.4f}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == '__main__':
    print_results()
    print("="*80)
    print_results('benchmark_results_llama-3.3-70b-versatile_clean.csv')
    print("="*80)
    print_results("benchmark_results_moonshotai/kimi-k2-instruct-0905_clean.csv")
    print("="*80)
    print_results("benchmark_results_meta-llama/llama-4-maverick-17b-128e-instruct_clean.csv")
    print("="*80)
    print("llama-maverick COD")
    print_results("benchmark_results_meta-llama/llama-4-maverick-17b-128e-instruct_COD.csv")
    print("="*80)
    print_results('benchmark_results_llama-3.1-8b-instant_clean.csv')
