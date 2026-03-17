import os, re
import random
import pandas as pd
import numpy as np
import ast


def remove_go_comments(source: str) -> str:
    """This function removes all the comments from a go file"""
    pattern = re.compile(
        r"""
        ("(?:\\.|[^"\\])*")        |  # double-quoted strings
        (`[^`]*`)                  |  # raw string literals
        (//[^\n]*$)                |  # line comments
        (/\*[\s\S]*?\*/)              # block comments
        """,
        re.MULTILINE | re.VERBOSE,
    )

    def replacer(match):
        # If it's a string literal, keep it
        if match.group(1) or match.group(2):
            return match.group(0)
        # Otherwise it's a comment → remove
        return ""

    return re.sub(pattern, replacer, source)

def remove_comments_from_all_benchmark(benchmarks_path: str):
    """The function remove all comments from the benchmarks found in benchmarks_path"""
    benchmarks_folder = os.path.join(os.curdir, benchmarks_path)
    for type in os.listdir(benchmarks_folder):
        # blocking or nonblocking
        type_folder = os.path.join(benchmarks_folder, type)    
        for proj in os.listdir(type_folder):
            proj_folder = os.path.join(type_folder, proj)

            for fragment in os.listdir(proj_folder):
                frag_path = os.path.join(proj_folder, fragment)
                frag_path = os.path.join(frag_path, proj+fragment+"_test.go")
                with open(frag_path,"r+") as f:
                    content = f.read()
                    content = remove_go_comments(content)
                    f.seek(0)
                    f.write(content)
                    f.truncate()


def process_results_csv(model_name : str = 'qwen/qwen3-32b'):
    file_name = f'benchmark_results_{model_name}.csv'
    df = pd.read_csv(file_name)
    df['classification'] = df['classification'].apply(ast.literal_eval)
    # .apply(pd.Series) turns keys into column names and values into row data
    classification_cols = df['classification'].apply(pd.Series)

    # Merge back and drop the original column
    df = pd.concat([df.drop(['classification'], axis=1), classification_cols], axis=1)

    df = df.replace('None', np.nan)

    # Remove leading numbers from 'subsubtype' (e.g., "1.2 ", "2.3 ")
    # Regex explanation:
    # ^      : Start of string
    # \d+    : One or more digits
    # \.     : A literal dot
    # \d+    : One or more digits
    # \s+    : One or more whitespace characters
    df['subsubtype'] = df['subsubtype'].str.replace(r'^\d+\.\d+\s+', '', regex=True)
    
    df.to_csv(f'benchmark_results_{model_name}_clean.csv', index=False)

    print(df.head())


def collect_instances(root_dir):
    
    instances = []

    for class_name in ["blocking", "nonblocking"]:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for project in os.listdir(class_path):
            project_path = os.path.join(class_path, project)
            if not os.path.isdir(project_path):
                continue

            for case_id in os.listdir(project_path):
                case_path = os.path.join(project_path, case_id)
                if not os.path.isdir(case_path):
                    continue

                # Expected file name: projectnumber_test.go
                expected_file = f"{project}{case_id}_test.go"
                file_path = os.path.join(case_path, expected_file)

                if os.path.isfile(file_path):
                    instances.append(file_path)
                else:
                    # fallback: pick any *_test.go file in the folder
                    for f in os.listdir(case_path):
                        if f.endswith("_test.go"):
                            instances.append(os.path.join(case_path, f))
                            break

    return instances


def create_validation_test(root_dir : str, val_size : int, test_file :str, val_file):
    random.seed(42)  

    instances = collect_instances(root_dir)

    if len(instances) < val_size:
        raise ValueError("Dataset smaller than validation size.")

    # Shuffle and split
    random.shuffle(instances)
    val_set = instances[:val_size]
    test_set = instances[val_size:]

    with open(val_file, "w") as f:
        for item in val_set:
            f.write(item + "\n")

    with open(test_file, "w") as f:
        for item in test_set:
            f.write(item + "\n")

    print(f"Total instances: {len(instances)}")
    print(f"Validation set: {len(val_set)}")
    print(f"Test set: {len(test_set)}")


if __name__ == '__main__':
    remove_comments_from_all_benchmark("benchmarks/goker")
    create_validation_test(
        root_dir="benchmarks/goker", 
        val_size=20, 
        test_file="benchmarks_paths/test_set.txt", 
        val_file="benchmarks_paths/validation_set.txt"
    ) 