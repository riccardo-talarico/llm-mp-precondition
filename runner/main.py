from agent.chainOfDebug import ChainOfDebugAgent
from agent.verificationAgent import VerificationAgent
from utils.experiments import *
from pathlib import Path
from utils.results import print_results, print_token_count
import os
from datetime import datetime


cfg_path = os.getenv("CONFIG_PATH","config/experiment.yaml")
cfg = load_config(Path(cfg_path))
#DEBUG:
print(f"DEBUG: Model={cfg.model}, Base={cfg.base_url}",flush=True)
print("Starting warm-up",flush=True)
do_warmup(cfg.base_url, cfg.model, 30, cfg.options)
print("Warm-up finished",flush=True)

logger = ExperimentLogger()


print("Initializing agents.", flush=True)
chain_agent = ChainOfDebugAgent('Ollama', cfg.model, ollama_cfg=cfg, debug_level=3)
no_chain_agent = VerificationAgent(cfg.paths_file,cfg.paths_file,'Ollama', cfg.model, ollama_cfg=cfg)


chain_agent.compile_chain()

# Running Chain agent
print(f"Running Chain Agent", flush=True)
df_chain, config_chain = chain_agent.run_on_benchmark(cfg.paths_file, save_usage_metadata=True)
df_chain = df_chain.T
chain_id = logger.log_run(config_chain, df_chain)

# Running No-Chain agent
print(f"Running No Chain Agent", flush=True)
df_no_chain, config_no_chain = no_chain_agent.run_on_benchmark(validation=True,save_usage_metadata=True,insert_sleep=False)
df_no_chain = df_no_chain.T
no_chain_id = logger.log_run(config_no_chain, df_no_chain)

# Printing results
print("="*60)
print("CHAIN AGENT:")
print(f"Time required: {chain_agent.last_run_time}")
print(f"DEBUG: usage_metadata: {chain_agent.usage_metadata}")
input_tokens, output_tokens = config_chain["input_tokens"], config_chain["output_tokens"]
print(f"Input token: {input_tokens}, output tokens:{output_tokens}.\nTotal tokens:{input_tokens+output_tokens}")
print_results(f"results/classifications/run_{chain_id}.csv", cfg.paths_file)

print("="*60)
print("NO CHAIN AGENT:")
print(f"Time required: {no_chain_agent.last_run_time}")
print_token_count(no_chain_agent.usage_metadata)
print_results(f"results/classifications/run_{no_chain_id}.csv", cfg.paths_file)
print("="*60)
