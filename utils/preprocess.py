import os, re, sys
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
    for proj in os.listdir(benchmark_folder):
        proj_folder = os.path.join(benchmark_folder, proj)
        for fragment in os.listdir(proj_folder):
            readme_path = os.path.join(proj_folder,fragment+"/README.md")
            with open(readme_path, "r") as f:
                rdme = f.read()
                readmes.append(rdme)
    return readmes

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

def create_benchmark_labels():
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

def remove_comments_from_all_benchmark():
    """The function remove all comments from the benchmark"""
    benchmarks_folder = os.path.join(os.curdir, "gomela/benchmarks/blocking")
    for proj in os.listdir(benchmarks_folder):
        proj_folder = os.path.join(benchmarks_folder, proj)

        for fragment in os.listdir(proj_folder):
            frag_path = os.path.join(proj_folder, fragment)
            frag_path = os.path.join(frag_path, proj+fragment+"_test.go")
            with open(frag_path,"r+") as f:
                content = f.read()
                content = remove_go_comments(content)
                f.seek(0)
                f.write(content)
                f.truncate()



if __name__ == '__main__':
    #create_benchmark_labels()
    remove_comments_from_all_benchmark()
                
    
    

    
                
                
                
            
    