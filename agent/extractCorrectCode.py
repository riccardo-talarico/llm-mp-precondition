from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage

from typing_extensions import Literal
from pydantic import BaseModel, Field

from utils.tool_analysis import log_tool_interactions

import os, time, sys, re
import subprocess, tempfile, shutil
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

output_file = "./unprocessed_goker/fixes.jsonl"


extraction_prompt = """
You are a software repair assistant specialized in Go programs.

You are given:
1. A buggy Go code snippet
2. A README describing the bug, including bug classification and possibly a backtrace

Your task is to produce a patch that fixes the bug.

Instructions:
- Carefully read the README and understand the bug cause.
- Use the backtrace to locate the buggy lines.
- Modify the code to fix the root cause, not just the symptom.
- Do not rewrite the entire program; produce a minimal fix.
- The output must be a valid git diff.
- If multiple fixes are possible, choose the most likely one based on the README.
- If the README strongly implies a fix, follow it.
- If the fix is uncertain, still propose the most reasonable fix.

Respond following the output schema.

Code: 
{code}

README: 
{readme}
"""

evaluation_prompt = """
You are a code review assistant evaluating bug fixes for Go programs.

You are given:
1. The buggy code
2. The README describing the bug and backtrace
3. A proposed patch (git diff)
4. An explanation of the fix

Your task is to evaluate whether the fix is likely correct and assign a confidence level.

Evaluation guidelines:
- Very High confidence:
  The README explicitly describes the fix and the patch implements it.
- High confidence:
  The README clearly describes the bug cause and the patch fixes that cause.
- Medium confidence:
  The bug is described and the fix is reasonable but not explicitly stated.
- Low confidence:
  The fix is speculative or multiple fixes are possible.
- Very Low confidence:
  The fix does not clearly address the bug or may be incorrect.

Also evaluate:
- Does the patch modify the lines indicated by the backtrace?
- Does the patch address the bug category described in the README?
- Could the bug still occur after this fix?
- Are there alternative fixes?

Respond following the output schema.
Code:
{code}

README:
{readme}

Fix (source, reasoning and diff):
{fix}
"""

class Extraction(BaseModel):
    fix_source: Literal[
        'explicit_readme_fix',
        'readme_bug_description',
        'code_comment',
        'inferred_from_code',
        'guess'
    ]

    reasoning: str = Field(
        description="Why the fix is correct and why this confidence level was assigned."
    )

    diff: str = Field(
        description="Git diff that fixes the bug."
    )

class Evaluation(BaseModel):
    confidence: Literal[
        'very_high',
        'high',
        'medium',
        'low',
        'very_low'
    ] = Field(description='Evaluation of the reliability of the fix, based on the information provided')
    reasoning: str = Field(description='Why you assigned that confidence value')
    remaining_risks :str = Field(description='Any remaining risks if applying the fix, if any')

class ExtractorAgent():
    """LLM agent to extract a fix for some of the goker snippets of code.
    The extraction must be solely based on the information present in the README and the comments of the code.
    A correction based uniquely on an LLM guess is not accepted."""

    def __init__(self, extractor_provider : str, extractor_model: str, evaluator_provider : str, evaluator_model : str):
        self.extractor_model = extractor_model
        self.evaluator_model = evaluator_model
        self.extractor = self.parse_provider(extractor_provider, extractor_model)
        self.extractor = self.extractor.with_structured_output(Extraction)

        self.evaluator = self.parse_provider(evaluator_provider, evaluator_model)
        self.evaluator = self.evaluator.with_structured_output(Evaluation)
        
    def parse_provider(self, provider: str, model: str):
        if provider == 'Google':
            api_key = os.getenv("GEMINI_API_KEY")
            return ChatGoogleGenerativeAI(model=model,google_api_key=api_key, max_retries = 5)
        elif provider == 'Groq':
            api_key = os.getenv('GROQ_API_KEY')
            return ChatGroq(model=model, groq_api_key=api_key, max_retries = 5)
        else:
            print("Other providers not currently supported")
            return None 

    def run(self, folder:str = "./unprocessed_goker/", output_path :str = output_file):
        """The function makes the agent run on the entire unprocessed_goker, to extract a possible .diff file for each snippet of code.
        The possible fix and the evaluation are iteratively dumped in a JSONL file, stored in 'output_path'.
        This has the default value of ./unprocessed_goker/fixes.jsonl"""
        for cls in ['blocking','nonblocking']:
            clspath = os.path.join(folder,cls)
            for proj in os.listdir(clspath):
                projpath = os.path.join(clspath,proj)
                for case in os.listdir(projpath):
                    casepath = os.path.join(projpath,case)
                    expected_file = f"{proj}{case}_test.go"
                    file_path = os.path.join(casepath, expected_file)
                    readme_path = os.path.join(casepath,"README.md")

                    with open(file_path, "r") as f:
                        code = f.read()
                    with open(readme_path, "r") as f:
                        readme = f.read()
                    print(f"Fixing instance: {expected_file}")
                    try:
                        fix = self.extractor.invoke([SystemMessage(extraction_prompt.format(code=code, readme=readme))])
                        log_tool_interactions(fix)
                        evaluation = self.evaluator.invoke([SystemMessage(evaluation_prompt.format(code=code,readme=readme,fix=fix))])
                        print(evaluation)
                        log_tool_interactions(evaluation)
                        record = {
                            "models":(self.extractor_model, self.evaluator_model),
                            "instance_id": expected_file,
                            "fix": fix.model_dump(),
                            "evaluation": evaluation.model_dump(),
                            "error": None
                        }
                    except Exception as e:
                        print(f"An error occurred during fix of {expected_file}. This code instance will be skipped. {e}")
                        proceed = input("Do you want to continue? [Y/N]")
                        if proceed == 'N':
                            return
                        record = {
                            "models":(self.extractor_model, self.evaluator_model),
                            "instance_id": expected_file,
                            "fix": None,
                            "evaluation": None,
                            "error": str(e)
                        }
                    with open(output_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print("-"*20+" Sleep inserted to avoid consuming all tokens "+ "-"*20)
                    time.sleep(10)

import json

def load_and_filter(jsonl_path):
    confidence_rank = {
        "very_low": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "very_high": 4
    }

    allowed = {"high", "very_high"}

    best_instances = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)

            if data["fix"] is None or data["evaluation"] is None:
                continue

            instance_id = data["instance_id"]
            confidence = data["evaluation"]["confidence"]

            if confidence not in allowed:
                continue

            if instance_id not in best_instances:
                best_instances[instance_id] = data
            else:
                current_conf = confidence_rank[
                    best_instances[instance_id]["evaluation"]["confidence"]
                ]
                new_conf = confidence_rank[confidence]
                # Store the one with the highest confidence
                if new_conf > current_conf:
                    best_instances[instance_id] = data
                

    filtered_list = list(best_instances.values())
    total_instances = len(filtered_list)

    return filtered_list, total_instances   

def find_original_file_path(folder, id):
    for cls in ['blocking','nonblocking']:
        cls_path = Path(folder).joinpath(cls)
        
        proj_name,case_number = "",""
        digit_found = False
        for c in id:
            if c.isdigit():
                digit_found = True
                case_number+=c
            if not digit_found:
                proj_name += c       
 
        possible_path = cls_path.joinpath(Path(proj_name+"/"+case_number+"/"+id))
        if possible_path.exists():
            return possible_path



def is_valid_unified_diff(diff_text: str) -> bool:
    if not diff_text:
        return False

    lines = diff_text.splitlines()

    has_old = any(line.startswith('--- ') for line in lines)
    has_new = any(line.startswith('+++ ') for line in lines)
    has_hunk = any(line.startswith('@@ ') for line in lines)
    has_changes = any(line.startswith('+') or line.startswith('-') for line in lines)

    return has_old and has_new and has_hunk and has_changes

        
def normalize_diff(diff_text, filename):
    lines = diff_text.splitlines()

    new_lines = []
    for line in lines:
        if line.startswith('--- '):
            new_lines.append(f'--- {filename}')
        elif line.startswith('+++ '):
            new_lines.append(f'+++ {filename}')
        else:
            new_lines.append(line)

    return '\n'.join(new_lines) + '\n'


import re

def apply_unified_diff(original_code: str, diff_text: str) -> str:
    original_lines = original_code.splitlines()
    patched_lines = original_lines.copy()

    diff_lines = diff_text.splitlines()
    
    hunks = []
    current_hunk = []

    for line in diff_lines:
        if line.startswith('@@'):
            if current_hunk:
                hunks.append(current_hunk)
                current_hunk = []
        if line.startswith(('@@', '+', '-', ' ')):
            current_hunk.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    offset = 0

    for hunk in hunks:
        # Extract hunk lines ignoring header
        hunk_body = [l for l in hunk if not l.startswith('@@')]

        # Build pattern to find where to apply patch
        context_lines = [l[1:] for l in hunk_body if l.startswith(' ')]
        remove_lines = [l[1:] for l in hunk_body if l.startswith('-')]
        add_lines = [l[1:] for l in hunk_body if l.startswith('+')]

        # Find context in patched_lines
        for i in range(len(patched_lines)):
            if patched_lines[i:i+len(context_lines)] == context_lines:
                start = i
                break
        else:
            raise RuntimeError("Could not find context for hunk")

        # Apply changes
        idx = start
        for line in hunk_body:
            if line.startswith(' '):
                idx += 1
            elif line.startswith('-'):
                patched_lines.pop(idx)
            elif line.startswith('+'):
                patched_lines.insert(idx, line[1:])
                idx += 1

    return "\n".join(patched_lines) + "\n"




def apply_diff_and_get_fixed_code(original_file_path, diff_text):

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy original file into temp directory
        temp_file = tmpdir / original_file_path.name
        shutil.copy(original_file_path, temp_file)

        # Write diff to temp file
        diff_file = tmpdir / "patch.diff"
        with open(diff_file, "w") as f:
            f.write(diff_text)

        # Consider different commands options
        commands = [
            ["patch", "-p0", "--fuzz=3", "-i", str(diff_file)],
            ["patch", "-p1", "--fuzz=3", "-i", str(diff_file)],
            ["patch", "-p0", "-l", "-i", str(diff_file)],  # ignore whitespace
        ]

        # Apply patch
        
        for cmd in commands:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True
            )

            # Read fixed code
            if result.returncode == 0:
                with open(temp_file, "r") as f:
                    fixed_code = f.read()
                    return fixed_code
            else:
                print(f"Stdout:{result.stdout}")
                print(f"Stderr:{result.stderr}")
                
        
        print(f"[WARNING] Patch failed.")


if __name__=='__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['extract','report','save']:
        print("Usage: python3 extractCorrectCode [extract/report/save]")
        exit(1)

    if sys.argv[1] == 'extract':
        a = ExtractorAgent(
            extractor_provider='Groq', 
            extractor_model='llama-3.1-8b-instant', 
            evaluator_provider='Groq', 
            evaluator_model="moonshotai/kimi-k2-instruct-0905"
            )
        a.run()
    elif sys.argv[1] == 'report':
        filtered_instances, count = load_and_filter(output_file)
        print(f"Count: {count}")
    else:
        extracted_path = "benchmarks/extracted/"
        extracted_path = Path(extracted_path) 
        extracted_path.mkdir(exist_ok=True)

        filtered_instances, _  = load_and_filter(output_file)
        for instance in filtered_instances:
            id = instance['instance_id']
            path = extracted_path.joinpath(Path(id+"/"))
            path.mkdir(exist_ok=True)
            readme = {
                'Models': instance['models'],
                'Fix Source': instance['fix']['fix_source'],
                'Fix Reasoning': instance['fix']['reasoning'],
                'Confidence Level': instance['evaluation']['confidence'],
                'Evaluation Reasoning': instance['evaluation']['reasoning'],
                'Remaining Risks':instance['evaluation']['remaining_risks']
            }

            with open(path.joinpath(Path("README.json")),"w+") as f:
                json.dump(readme, f, indent=4, sort_keys=True)                

            diff = instance['fix']['diff']
            original_file_path = find_original_file_path("unprocessed_goker", id)            
            diff = normalize_diff(diff, original_file_path.name)
            with open(original_file_path, "r") as f:
                original_code = f.read()
            if is_valid_unified_diff(diff):
                code = apply_unified_diff(original_code, diff)
                print(code)
                #TODO: the commands in apply_diff_and_get_fixed_code still fail, try to fix it
            break