import pandas as pd
import json, sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from langchain.messages import AIMessage


def get_usage_metadata(response, msg_id:int = 0):
    """The function tries to extract the usage_metadata from the 'response'.
    The couple '(msg_id, response.usage_metadata)' is appended to the result value."""
    res = []
    if isinstance(response, dict):
      try:
        raw = response['raw']
      except Exception as e:
        print(f"Unable to fetch the response. {e}")
        return res
      
    if isinstance(raw,list):
      for msg in raw:
        if isinstance(msg, AIMessage): 
          if hasattr(msg, 'usage_metadata'):
            res.append((msg_id,msg.usage_metadata))
          elif hasattr(msg,'response_metadata'):
            res.append((msg_id,msg.response_metadata))
          elif hasattr(msg,'additional_kwargs'):
            res.append((msg_id,msg.additional_kwargs))
    
    elif isinstance(raw,dict):
      try:
        res.append((msg_id, raw['response_metadata']))
      except Exception as e:
        print(f"Could not append metadata: {e}")
    elif isinstance(raw,AIMessage):
      try:
        res.append((msg_id, raw.response_metadata))
      except Exception as e:
        print(f"Could not append metadata: {e}")
    
    
    return res


def try_into_dataframe(data, model_name):
    """The function tries to save the data into a dataframe. 
    In case the operation fails, the data is saved into a json file.
    The save file path is 'result/model_name'"""
    try:
        res = pd.DataFrame(data)
    except Exception as e:
        print(f"Error while transforming into dataframe: {e}")
        res = pd.DataFrame()
        with open("results/"+model_name+".json", "a+") as f:
            f.write(json.dumps(data, indent=4))
    return res


def get_token_count(usage_metadata):
    """The function tries to fetch the 'usage_metadata' to extract the input and output tokens and returns them.
    In case the extraction fails it returns (-1,-1)."""
    if usage_metadata == []:
      return (0,0)
    input_tokens,output_tokens = 0,0
    for id,metadata in usage_metadata:
      try:
        usage = metadata.get("token_usage", {}) # Try generic name first
        if not usage:
            # Fallback to Ollama key names
            input_tokens += metadata.get("prompt_eval_count", 0)
            output_tokens += metadata.get("eval_count", 0)
        else:
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
      except Exception as e:
        print(f"Error during extraction of token count: {e}")
        return (0,0)
    return (input_tokens,output_tokens)

def print_token_count(usage_metadata):
    """The function prints the input, output and total tokens usage extracted from the 'usage_metadata'.
    In case there is an error during the extraction it immediately returns, without printing anything."""
    input_tokens,output_tokens=get_token_count(usage_metadata)
    if input_tokens == -1:
      return
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens: {input_tokens+output_tokens}")


def hierarchical_accuracy(df):
    score = 0.0

    for _, row in df.iterrows():
        #print(f"True:{row['cls_true']}")
        #print(f"Pred:{row['cls_pred']}")
        if row['cls_true'] == row['cls_pred']:
            if row["type_true"].lower() == row["type_pred"].lower():
                if row["subtype_true"].lower() == row["subtype_pred"].lower():
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
        blocking = {k: add_class('Blocking',items[k]) for k in ids if k in items}
    with open("benchmarks/configures/goker/nonblocking.json", "r") as f:
        items = json.load(f)
        nonblocking = {k: add_class('Nonblocking',items[k]) for k in ids if k in items}
    df = pd.DataFrame(blocking | nonblocking)
    return df.T

def print_results(file_name_results: str, paths_file :str):
    df_results = pd.read_csv(file_name_results)
    df_true = extract_true_df(paths_file = paths_file)
    total_prgs = len(df_true)

    for col in df_results.columns:
        df_results[col] = df_results[col].fillna('None')
        df_results[col] = df_results[col].astype(str)
    
    df_results.index = df_true.index
    df = df_true.join(df_results, lsuffix='_true', rsuffix='_pred')
    y_true = df_true['cls'].str.strip()
    try:
        y_pred = df_results['cls'].str.strip()
    except:
       y_pred = df_results['class'].str.strip()
    
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

def set_to_path(set: str):
    if 'validation' in set:
        paths_file = "benchmarks_paths/validation_set.txt"
    elif 'test' in set:
        paths_file = "benchmarks_paths/test_set.txt"
    else:
        print(f"Invalid set name {set}, must contain validation or set, returning None")
        paths_file = None
    return paths_file


def get_experiments_clusters(
      index_path:str="results/master_index.csv",
      matching_keys:list[str] = ['model','temperature','architecture','set','prompt_v','top_k','top_p'],
      filters: dict | None = None
    ):
    index = pd.read_csv(index_path)
    
    if filters:
        for column, value in filters.items():
            if column in df.columns:
                # Handle cases where value might be a list (e.g., multiple models)
                if isinstance(value, list):
                    df = df[df[column].isin(value)]
                else:
                    df = df[df[column] == value]
            else:
                print(f"Warning: Column '{column}' not found in CSV. Skipping filter.")
    

    grouped = index.groupby(matching_keys)

    clusters = []
    for specs, frame in grouped:
       clusters.append({
        "hyperparameters": dict(zip(matching_keys, specs if isinstance(specs, tuple) else [specs])),
        "run_ids": frame['run_id'].tolist(),
        "output_files": frame['output_file'].tolist(),
        "time": np.array(frame['time']),
        'input_tokens': np.array(frame['input_tokens']),
        'output_tokens': np.array(frame['output_tokens']),
        'seeds' : frame['seed'],
        "count": len(frame)
        })
    
    return clusters


def compute_cluster_metrics(cluster):
    paths_file = set_to_path(cluster['hyperparameters']['set'])
    df_true = extract_true_df(paths_file = paths_file)
    f1 = []
    hacc = []
    acc = []
    for output in cluster['output_files']:
        df_results = pd.read_csv("results/classifications/"+output)
        total_prgs = len(df_true)

        for col in df_results.columns:
            df_results[col] = df_results[col].fillna('None')
            df_results[col] = df_results[col].astype(str)

        df_results.index = df_true.index
        df = df_true.join(df_results, lsuffix='_true', rsuffix='_pred')
        y_true = df_true['cls'].str.strip()
        try:
            y_pred = df_results['cls'].str.strip()
        except:
           y_pred = df_results['class'].str.strip()

        f1.append(f1_score(y_true, y_pred, average='weighted'))
        acc.append(balanced_accuracy_score(y_true, y_pred))
        hacc.append(hierarchical_accuracy(df))
    
    f1 = np.array(f1)
    acc = np.array(acc)
    hacc = np.array(hacc)
    results= {
        'f1': (f1.mean(), f1.std()), 
        'balanced_accuracy':(acc.mean(), acc.std()), 
        'hierarchical_accuracy':(hacc.mean(),hacc.std()),
        'time': (cluster['time'].mean(), cluster['time'].std()),
        'input_tokens': (cluster['input_tokens'].mean(), cluster['input_tokens'].std()),
        'output_tokens': (cluster['output_tokens'].mean(), cluster['output_tokens'].std()),
        'seeds':cluster['seeds']
       }
    return results
    
    



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
    #df=extract_true_df(paths_file)
    #print_results(f"results/chain_llama3.1:8b_2026-04-21_08-01-36.csv", paths_file)
    #print_results(f"results/no_chain_llama3.1:8b_2026-04-21_08-16-07.csv", paths_file)
    clusters = get_experiments_clusters()
    for cluster in clusters:
       results = compute_cluster_metrics(cluster)
       print("="*40)
       print(f"Model: {cluster['hyperparameters']['model']}, Prompt version: {cluster['hyperparameters']['prompt_v']}")
       print(f"Number of experiments: {cluster['count']}")
       print(f"Temperature: {cluster['hyperparameters']['temperature']}")
       print(f"Avg time: {cluster['time'].mean()}, std:{cluster['time'].std()}")
       print(f"Avg input tokens: {cluster['input_tokens'].mean()}, std:{cluster['input_tokens'].std()}")
       print(f"Avg output tokens: {cluster['output_tokens'].mean()}, std:{cluster['output_tokens'].std()}")
       print(f"F1:{results['f1'][0]}, std: {results['f1'][1]}")
       print(f"Balanced Accuracy:{results['balanced_accuracy'][0]}, std: {results['balanced_accuracy'][1]}")
       print(f"Hierachical Accuracy:{results['hierarchical_accuracy'][0]}, std: {results['hierarchical_accuracy'][1]}")


 

