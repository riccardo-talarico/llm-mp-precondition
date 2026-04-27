import os, time
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage

from utils.graph import BugClassification
from utils.output_parser import try_to_invoke
from utils.tool_analysis import log_tool_interactions
from utils.experiments import OllamaExperimentConfig, ExperimentLogger, extract_latest_prompt_version, get_prompt_version
from utils.results import extract_id, get_usage_metadata, try_into_dataframe, print_token_count


class VerificationAgent():

  def __init__(
      self, 
      validation_paths: str, 
      test_paths : str,
      provider='Google',
      model="gemini-2.5-flash", 
      logging = True,
      ollama_cfg : None | OllamaExperimentConfig = None,
      prompt_version : str | None = None
      ):
    if provider == 'Google':
      api_key = os.getenv("GEMINI_API_KEY")
      self.llm = ChatGoogleGenerativeAI(model=model,google_api_key=api_key, max_retries = 3)
    elif provider == 'Groq':
      api_key = os.getenv('GROQ_API_KEY')
      self.llm = ChatGroq(model=model, groq_api_key=api_key, max_retries=1)
    elif provider == 'Ollama':
      self.llm = ChatOllama(
          base_url=ollama_cfg.base_url,
          model=ollama_cfg.model,
          temperature=ollama_cfg.temperature,
          num_ctx=ollama_cfg.options.get("num_ctx", 4096),
          top_p=ollama_cfg.options.get("top_p", 0.9),
          top_k=ollama_cfg.options.get("top_k", 40),
          seed = ollama_cfg.seed,
      )
      self.cfg = ollama_cfg
    else:
      print("Other providers not currently supported")
      self.llm = None  
    self.validation_paths = validation_paths
    self.test_paths = test_paths
    self.logging = logging
    self.model = model
    self.usage_metadata = []
    self.provider = provider
    self.last_run_time = -1
    self.prompt_v = prompt_version if prompt_version else extract_latest_prompt_version("detection_and_classification")
    self.structured_llm = self.llm.with_structured_output(BugClassification, include_raw=True)
      
  def get_config(self):
    if self.provider == 'Ollama':
      return {
        "architecture": "CoT",
        "set": self.cfg._set,
        "model": self.model,
        "temperature": self.cfg.temperature,
        "top_p":self.cfg.options["top_p"],
        "top_k":self.cfg.options["top_k"],
        "num_ctx": self.cfg.options["num_ctx"],
        "seed": self.cfg.seed,
        "prompt_v": self.prompt_v
      }
    else:
      return {
        "architecture": "CoT",
        "model": self.model,
        "temperature": self.llm.temperature,
        "top_p": getattr(self.llm, "top_p", None),
        "top_k":getattr(self.llm, "top_k",None),
        "num_ctx": getattr(self.llm, "num_ctx", None),
        "seed" : getattr(self.llm, "seed", None),
        "prompt_v": self.prompt_v
        }

    
  def run_on_benchmark(self, validation : bool = False, save_usage_metadata : bool=True, insert_sleep:bool=True):
    
    path = self.validation_paths if validation else self.test_paths
    with open(path,'r') as f:
      prg_paths = f.readlines()

    prompt = get_prompt_version("detection_and_classification", self.prompt_v)
    sysMsg = SystemMessage(content=prompt)
    classification_data = {}
    thinking_log = {}
    verified_prg = 0
    start = time.time()
    for prg_path in prg_paths:
      with open(prg_path[:-1], "r") as f:
        id = extract_id(prg_path[:-1])
        print(f"Id: {id}")
        prog = f.read()
        messages = [sysMsg, HumanMessage(content=prog)]
        default_response = {
          'cls':'None','type': 'None', 'subtype': 'None'
        }
        response = try_to_invoke(self.structured_llm,messages,BugClassification,self.llm, default_response)
        try:
          print(f"{response['parsed']}")
          print(f"{response['raw']}")
          classification_data[id]= response['parsed'].get_classification()
        except:
          print(response)
          response = {'parsed':response, 'raw':response}
          classification_data[id] = response['parsed']
        try:
          reasoning = response['raw'].additional_kwargs.get("reasoning_content")
          thinking_log[id] = reasoning
        except Exception as e:
          print(f"Cannot extract reasoning tokens: {e}")

        
        verified_prg+=1
        
        if save_usage_metadata:
          self.usage_metadata += get_usage_metadata(response, verified_prg-1)

        print(f"Progress: {verified_prg}/{len(prg_paths)}")
        if insert_sleep:
          print("-"*20+" Sleep inserted to avoid consuming all tokens "+"-"*20)
          time.sleep(15)
    self.last_run_time = time.time()-start

    if self.logging:
      log_tool_interactions(response)

    res = try_into_dataframe(classification_data, self.model)

    config = self.get_config()
    config["set"] = "validation" if validation else "test"

    return res, config
  
    
#Groq models: qwen/qwen3-32b, llama-3.3-70b-versatile, llama-3.1-8b-instant, meta-llama/llama-4-maverick-17b-128e-instruct,
# moonshotai/kimi-k2-instruct-0905
if __name__=='__main__':
  a = VerificationAgent(
    provider='Groq',
    model='llama-3.1-8b-instant',
    test_paths='benchmarks_paths/test_set.txt', 
    validation_paths='benchmarks_paths/validation_set.txt'
    )
  df, config = a.run_on_benchmark(validation=True, save_usage_metadata=True, insert_sleep=True)
  print(f"Time required: {a.last_run_time}")
  print(f"Usage metadata: {a.usage_metadata}")
  print_token_count(a.usage_metadata)
  df = df.T
  logger = ExperimentLogger()
  logger.log_run(config, df)