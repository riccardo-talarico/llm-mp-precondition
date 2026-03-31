import os, time, json
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage

from utils.graph import BugClassification
from utils.output_parser import try_to_invoke
from utils.tool_analysis import log_tool_interactions
from utils.prompts import SINGLE_PROMPT_DETECTION_AND_CLASSIFICATION
from utils.results import extract_id, get_usage_metadata, try_into_dataframe, print_token_count


class VerificationAgent():

  def __init__(self, validation_paths: str, test_paths : str,provider='Google',model="gemini-2.5-flash", logging = True):
    if provider == 'Google':
      api_key = os.getenv("GEMINI_API_KEY")
      self.llm = ChatGoogleGenerativeAI(model=model,google_api_key=api_key, max_retries = 3)
    elif provider == 'Groq':
      api_key = os.getenv('GROQ_API_KEY')
      self.llm = ChatGroq(model=model, groq_api_key=api_key, max_retries=1)
    else:
      print("Other providers not currently supported")
      self.llm = None  
    self.validation_paths = validation_paths
    self.test_paths = test_paths
    self.logging = logging
    self.model = model
    self.usage_metadata = []
    self.last_run_time = -1
    self.structured_llm = self.llm.with_structured_output(BugClassification, include_raw=True)
      
  def run_on_benchmark(self, validation : bool = False, save_usage_metadata : bool=True, insert_sleep:bool=True):
    
    path = self.validation_paths if validation else self.test_paths
    with open(path,'r') as f:
      prg_paths = f.readlines()

    sysMsg = SystemMessage(content=SINGLE_PROMPT_DETECTION_AND_CLASSIFICATION)
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
          'class':'None','type': 'None', 'subtype': 'None'
        }
        response = try_to_invoke(self.structured_llm,messages,BugClassification,self.llm, default_response)
        try:
          print(f"{response['parsed']}")
          print(f"{response['raw']}")
        except:
          print(response)
          response = {'parsed':response, 'raw':response}
        try:
          reasoning = response['raw'].additional_kwargs.get("reasoning_content")
          thinking_log[id] = reasoning
        except Exception as e:
          print(f"Cannot extract reasoning tokens: {e}")

        classification_data[id]= response['parsed']
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

    res = try_into_dataframe(classification_data, a.model)

    return res
  
  def try_to_invoke(self, msg, default_response):
    while True:
      try:
        response = self.agent.invoke(msg)
        return response
      except Exception as e:
        print(f"Error while invoking the model {e}")
        s = input("Return default response? [Y/N]")
        if s == 'Y':
          return default_response
        

    
#Groq models: qwen/qwen3-32b, llama-3.3-70b-versatile, llama-3.1-8b-instant, meta-llama/llama-4-maverick-17b-128e-instruct,
# moonshotai/kimi-k2-instruct-0905
if __name__=='__main__':
  a = VerificationAgent(
    provider='Groq',
    model='llama-3.1-8b-instant',
    test_paths='benchmarks_paths/test_set.txt', 
    validation_paths='benchmarks_paths/validation_set.txt'
    )
  df = a.run_on_benchmark(validation=True, save_usage_metadata=True, insert_sleep=False)
  print(f"Time required: {a.last_run_time}")
  print(f"Usage metadata: {a.usage_metadata}")
  print_token_count(a.usage_metadata)
  stop = False
  while not stop:
    try:
      df.to_csv(f"results/benchmark_results_{a.model}.csv", index=True)      
      stop = True
    except Exception as e:
      stop = False
      print(f"Exception: {e}")
      i = input("Retry? [Y/N]")
      if i != "Y":
        stop = True
    


