from agent.chainOfDebug import ChainOfDebugAgent
from agent.verificationAgent import VerificationAgent
from utils.experiments import *
from pathlib import Path
from utils.results import print_results, print_token_count
import os,sys
from datetime import datetime


def run_full_benchmark(cfg):
    for model in cfg.models:
        agent = VerificationAgent(cfg.paths_file,"benchmarks_paths/full.txt",'Ollama',model, ollama_cfg=cfg)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df = agent.run_on_benchmark(validation=False,save_usage_metadata=True,insert_sleep=False)
        df = df.T
        df.to_csv(f"results/full_no_chain_{agent.model}_{timestamp}.csv", index=True)
        print("="*60)
        print(f"AGENT: {model}")
        print(f"Time required: {agent.last_run_time}")
        print_token_count(agent.usage_metadata)
        print_results(f"results/full_no_chain_{agent.model}_{timestamp}.csv", "benchmarks_paths/full.txt")
        print("="*60)


if __name__ == '__main__':


    try:
        full = sys.argv[1]
    except:
        full = False

    
    cfg_path = os.getenv("CONFIG_PATH","config/experiment.yaml")
    cfg = load_config(Path(cfg_path))
    #DEBUG:
    print(f"DEBUG: Model={cfg.model}, Base={cfg.base_url}")
    
    #DEBUG: for now testing the full results
    if full:
        run_full_benchmark(cfg)
    
    else:
        chain_agent = ChainOfDebugAgent('Ollama', cfg.model, ollama_cfg=cfg, debug_level=1)
        no_chain_agent = VerificationAgent(cfg.paths_file,cfg.paths_file,'Ollama',cfg.model, ollama_cfg=cfg)
        do_warmup(cfg.base_url, cfg.model, 30, cfg.options)

        chain_agent.compile_chain()

        # Running Chain agent
        chain_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df_chain = chain_agent.run_on_benchmark(cfg.paths_file, save_usage_metadata=True,insert_sleep=False)
        df_chain = df_chain.T
        df_chain.to_csv(f"results/chain_{chain_agent.model}_{chain_timestamp}.csv", index=False)

        # Running No-Chain agent
        no_chain_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df_no_chain = no_chain_agent.run_on_benchmark(validation=True,save_usage_metadata=True,insert_sleep=False)
        df_no_chain = df_no_chain.T
        df_no_chain.to_csv(f"results/no_chain_{no_chain_agent.model}_{no_chain_timestamp}.csv", index=True)

        # Printing results
        print("="*60)
        print("CHAIN AGENT:")
        print(f"Time required: {chain_agent.last_run_time}")
        print(f"DEBUG: usage_metadata: {chain_agent.usage_metadata}")
        print_token_count(chain_agent.usage_metadata)
        print_results(f"results/chain_{chain_agent.model}_{chain_timestamp}.csv", cfg.paths_file)
        print("="*60)
        print("NO CHAIN AGENT:")
        print(f"Time required: {no_chain_agent.last_run_time}")
        print_token_count(no_chain_agent.usage_metadata)
        print_results(f"results/no_chain_{no_chain_agent.model}_{no_chain_timestamp}.csv", cfg.paths_file)
        print("="*60)
