from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import os, yaml, requests, csv, json, uuid, re
from pydantic import BaseModel
import pandas as pd

@dataclass
class OllamaExperimentConfig:
    models : list[str]
    base_url: str
    model: str
    temperature: float
    seed: int
    _set : str
    warmup: bool
    paths_file: str
    options: dict[str, Any]
    prompt_config: dict[str, str]

    def extract_metadata(self):
        return {
            "model": self.model,
            'set': self._set, 
            "temperature": self.temperature, 
            "top_p": self.options.get("top_p"), 
            "top_k": self.options.get("top_k"),
            "num_ctx": self.options.get("num_ctx"),
            "prompt_config": self.options.get("prompt_config", default=None)
        }
    


def load_config(config_path: Path) -> OllamaExperimentConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg_url = raw.get("ollama_base_url")
    base = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434" or cfg_url
    return OllamaExperimentConfig(
        models = raw['models'],
        base_url=base.rstrip("/"),
        model=raw["model"],
        temperature=raw.get("temperature", 0.0),
        seed=raw.get("seed", 42),
        _set = raw.get("set","validation"),
        warmup=bool(raw.get("warmup", True)),
        options=raw.get("options", {}),
        paths_file=f'benchmarks_paths/{str(raw.get("set","validation"))}_set.txt',
        prompt_config=raw.get("prompt_config",{})
    )


def post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()

def do_warmup(base_url: str, model: str, timeout_sec: int, options: dict[str, Any]) -> None:
    """Function calls the model directly via HTTP, no langchain interface warmup.
    It reduces the first call latency."""
    payload = {
        "model": model,
        "prompt": "warmup",
        "stream": False,
        "options": options,
    }
    try:
        print(f"Calling: {base_url}/api/generate", flush=True)
        post_json(f"{base_url}/api/generate", payload, timeout_sec)
    except Exception as e:
        print(f"Warm-up failed for {model}: {e}")


class ExperimentLogger:
    def __init__(self, base_path: str = "results"):
        self.base_path = base_path
        self.classification_dir = os.path.join(base_path, "classifications")
        self.index_path = os.path.join(base_path, "master_index.csv")
        os.makedirs(self.classification_dir, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.index_path):
            with open(self.index_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "timestamp", "architecture", "set", "model", "input_tokens", "output_tokens", "temperature", "top_p", "top_k", "num_ctx","seed", "prompt_v", "output_file"])

    def log_run(self, metadata: Dict[str, Any], output_df: pd.DataFrame):
        """
        Saves the results of output_df into a csv and create a corresponding entry in the master_index
        using 'metadata'

        Args:
            metadata (Dict[str, Any]): metadata used to save the results, should contain information on architecture, prompt versioning and hyperparameters.
            output_df (pd.DataFrame): results of the agent run.
        
        Returns:
            str: the run_id created for saving the results
        """
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"run_{run_id}.csv"
        file_path = os.path.join(self.classification_dir, filename)

        # TODO: save also the raw outputs of the complete chain?

        # Saving only the classification
        output_df.to_csv(file_path, index=True)

        # Updates the searchable table
        with open(self.index_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id, 
                timestamp, 
                metadata.get("architecture"),
                metadata.get('set'),
                metadata.get("model"), 
                metadata.get("input_tokens", default=-1),
                metadata.get("output_tokens", default=-1),
                metadata.get("temperature"), 
                metadata.get("top_p"), 
                metadata.get("top_k"),
                metadata.get("num_ctx"),
                metadata.get("seed"),
                metadata.get("prompt_v"),
                filename
            ])
        
        print(f"Experiment {run_id} logged to {self.index_path}")
        return run_id


def get_prompt_version(prompt_name: str, version: str | None = None):
    directory = f"prompts/{prompt_name}"
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not version:
        version = extract_latest_prompt_version(prompt_name)

    for filename in os.listdir(directory):
        if version in filename and filename.endswith(('.yaml', '.yml')):
            return os.path.join(directory, filename)

    raise ValueError(f"Could not find version '{version}' in {directory}")


def extract_latest_prompt_version(prompt_name:str):
    directory = f"prompts/{prompt_name}"
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
   
    # Get all .yaml files with their full paths
    files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith(('.yaml', '.yml'))
    ]
    if not files:
        raise FileNotFoundError(f"No YAML files found in {directory}")

    # Get the file with the most recent modification time
    latest_file = max(files, key=os.path.getmtime)
    match = re.search(r"_v(\d+)", latest_file)
    return str(match.group(0))[1:]

def load_prompt_string(file_path: str) -> str:
    """
    Reads a YAML file and returns the raw template string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get("template", "")