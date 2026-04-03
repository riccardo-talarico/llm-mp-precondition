from dataclasses import dataclass
from typing import Any
from pathlib import Path
import os, yaml, requests

@dataclass
class OllamaExperimentConfig:
    base_url: str
    model: str
    temperature: float
    seed: int
    warmup: bool
    paths_file: str
    options: dict[str, Any]

def load_config(config_path: Path) -> OllamaExperimentConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg_url = raw.get("ollama_base_url")
    base = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434" or cfg_url
    return OllamaExperimentConfig(
        base_url=base.rstrip("/"),
        model=raw["model"],
        temperature=raw.get("temperature", 0.0),
        seed=raw.get("seed", 42),
        warmup=bool(raw.get("warmup", True)),
        options=raw.get("options", {}),
        paths_file=f'benchmarks_paths/{str(raw.get("set","validation"))}_set.txt'
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
        post_json(f"{base_url}/api/generate", payload, timeout_sec)
    except Exception as e:
        print(f"Warm-up failed for {model}: {e}")