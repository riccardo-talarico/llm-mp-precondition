import os
import pandas as pd


def process_readme(lines:str) -> tuple[str,str]:
    """The function takes as input the content of the readme file and extract the categorization of 
    the bug contained in the relative go program."""
    for line in lines:
        if "| Blocking |" in line:
            contents = line.split("|")
            return contents[5],contents[6]
            
def process_all_readme(benchmark_folder):
    """The function takes in input the benchmark folder to read all the readme files
    in order to return their content."""
    readmes = []
    for proj in os.listdir(benchmarks_folder):
        proj_folder = os.path.join(benchmarks_folder, proj)
        for fragment in os.listdir(proj_folder):
            readme_path = os.path.join(proj_folder,fragment+"/README.md")
            with open(readme_path, "r") as f:
                lines = f.readlines()
                readmes.append(lines)
    return readmes



if __name__ == '__main__':
    csv_path = os.path.join(os.curdir, "benchmark_classification.csv")
    bug_df = {"id":[], "subtype":[], "subsubtype":[]}
    benchmarks_folder = os.path.join(os.curdir, "gomela/benchmarks/blocking")
    for proj in os.listdir(benchmarks_folder):
        proj_folder = os.path.join(benchmarks_folder, proj)

        for fragment in os.listdir(proj_folder):
            bug_df["id"].append(proj+fragment) 
            
            readme_path = os.path.join(proj_folder,fragment+"/README.md")
            with open(readme_path, "r") as f:
                lines = f.readlines()
                subtype, subsubtype = process_readme(lines)
                bug_df["subtype"].append(subtype)
                bug_df["subsubtype"].append(subsubtype) 
    
    bug_df = pd.DataFrame(bug_df)
    bug_df.to_csv(csv_path)
                
                
                
            
    