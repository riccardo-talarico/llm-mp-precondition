import pandas as pd
import json, sys
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report


def hierarchical_accuracy(df):
    score = 0.0

    for _, row in df.iterrows():
        if row['cls_true'] == row['cls_pred']:
            if row["type_true"] == row["type_pred"]:
                if row["subtype_true"] == row["subtype_pred"]:
                    score += 1.0
                else:
                    score += (2/3)
            else:
                score += (1/3)

    return score / len(df)


def extract_id(prg_path : str) -> str:
    s = ""
    for c in prg_path[::-1][8:]:
        if c == "/":
            break
        s += c
    id = s[::-1]
    for i,c in enumerate(id):
        if c.isdigit():
            id = id[:i] + "_" + id[i:]
            break
    return id

def extract_true_df(paths_file: str) -> pd.DataFrame:
    ids = []
    with open(paths_file, "r") as f:
        lines = f.readlines()
        for prg_path in lines:
            id = extract_id(prg_path)
            ids.append(id[:-1])

    def add_class(cls:str, x:dict):
        x['cls'] = cls
        return x
    with open("benchmarks/configures/goker/blocking.json", "r") as f:
        items = json.load(f)
        blocking = {k: add_class('blocking',items[k]) for k in ids if k in items}
    with open("benchmarks/configures/goker/nonblocking.json", "r") as f:
        items = json.load(f)
        nonblocking = {k: add_class('nonblocking',items[k]) for k in ids if k in items}
    df = pd.DataFrame(blocking | nonblocking)
    return df.T

def print_results(file_name_results: str, paths_file :str):
    df_results = pd.read_csv(file_name_results)
    df_true = extract_true_df(paths_file = paths_file)
    total_prgs = len(df_true)

    for col in df_results.columns:
        df_results[col] = df_results[col].fillna('None')
        df_results[col] = df_results[col].astype(str)

    df = df_true.join(df_results, lsuffix='_true', rsuffix='_pred')

    y_true = df_true['cls'].str.strip()
    y_pred = df_results['cls'].str.strip()

    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    hacc = hierarchical_accuracy(df)

    missing_vals = (df_results['cls']=='None').sum()

    print(f"Missing values for {file_name_results}: {int(missing_vals)}/{total_prgs} = {missing_vals*100/total_prgs:.4f}%")
    print(f"Bugs correctly identified: {100-missing_vals*100/total_prgs:.4f}%")

    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy Score: {acc:.4f}")
    print(f"Hierarchical Accuracy Score: {hacc:.4f}")
    print("\nClassification Report:\n")
    print(report)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(f"Usage: utils/results.py [model_name] [validation/test]")

    model_name = sys.argv[1]
    set = sys.argv[2]

    if 'validation' in set:
        paths_file = "benchmarks_paths/validation_set.txt"
    elif 'test' in set:
        paths_file = "benchmarks_paths/test_set.txt"
    else:
        print(f"Invalid set name {set}, must contain validation or set")
        exit(1)
    df=extract_true_df(paths_file)
    print_results(f"results/benchmark_results_{model_name}.csv", paths_file)

 

