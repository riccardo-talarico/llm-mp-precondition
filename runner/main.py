from agent.chainOfDebug import ChainOfDebugAgent
from utils.experiments import *
from pathlib import Path
from utils.results import print_results
import pandas as pd
import os


if __name__ == '__main__':
    cfg_path = os.getenv("CONFIG_PATH","config/experiment.yaml")
    cfg = load_config(Path(cfg_path))
    cfg.base_url = os.getenv("OLLAMA_BASE_URL") or cfg.base_url
    
    
    agent = ChainOfDebugAgent('Ollama', cfg.model, ollama_cfg=cfg)
    do_warmup(cfg.base_url, cfg.model, 30, cfg.options)

    agent.compile_chain()
    df = agent.run_on_benchmark(cfg.paths_file)
    df = df.T
    df.to_csv(f"results/benchmark_results_{agent.model}.csv", index=False)
    print_results(f"results/benchmark_results_{agent.model}.csv", cfg.paths_file)
