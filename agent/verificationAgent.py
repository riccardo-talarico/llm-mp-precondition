import os, time, json
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from utils.tool_analysis import log_tool_interactions
from utils.prompts import SINGLE_PROMPT_DETECTION_AND_CLASSIFICATION
import pandas as pd


class VerificationAgent():

  def __init__(self, validation_paths: str, test_paths : str,provider='Google',model="gemini-2.5-flash", propose_fix=False, logging = True):
    if provider == 'Google':
      api_key = os.getenv("GEMINI_API_KEY")
      self.llm = ChatGoogleGenerativeAI(model=model,google_api_key=api_key, max_retries = 3)
    elif provider == 'Groq':
      api_key = os.getenv('GROQ_API_KEY')
      self.llm = ChatGroq(model=model, groq_api_key=api_key)
    else:
      print("Other providers not currently supported")
      self.llm = None  
    self.validation_paths = validation_paths
    self.test_paths = test_paths
    self.logging = logging
    self.model = model
    self.usage_metadata = []
    self.last_run_time = -1

    # Defining JSON response schema
    properties_schemas = {}
    properties_schemas['classification'] = {
      'type': "object",
      'description':"Provide a classification for the bug (if there is one) found in the code based on the hierarchy provided. In case there is no bug, put 'None' for all the required fields",
      'properties': {
        'class':{'type':'string','description':'Class of the bug (blocking or nonblocking)'},
        'type':{'type':'string','description':'Type of the bug, chosen according to the classification hierarchy'},
        'subtype':{'type':'string','description':'Subtype of the bug, chosen according to the classification hierarchy'}
      },
      'required':['class','subtype','type']
    }
    self.response_schema = {
      "type": "object",
      "description": "Response schema for the verification",
      "properties" : properties_schemas,
      'required': list(properties_schemas.keys())
    }
    if propose_fix:
      self.response_schema['properties']['proposed_fix']={
        'type': ['string','null'],
        'description':'Suggest eventual fixes to the program to make it satisfy all properties. Must be shown as git diffs.',
        'examples': ['-ch:=make(chan) +ch:=make(chan,1)']
      }
      self.response_schema['required'].append('proposed_fix')
    
    self.agent = self.llm.with_structured_output(self.response_schema, method="json_mode", include_raw=True)
      
  def run_on_benchmark(self, validation : bool = False, save_usage_metadata : bool=True):
    
    path = self.validation_paths if validation else self.test_paths
    with open(path,'r') as f:
      prg_paths = f.readlines()

    sysMsg = SystemMessage(content=SINGLE_PROMPT_DETECTION_AND_CLASSIFICATION)
    classification_data = {'id':[], 'classification': []}
    verified_prg = 0
    start = time.time()
    for prg_path in prg_paths:
      with open(prg_path[:-1], "r") as f:
        print(f"File: {prg_path[:-1]}")
        prog = f.read()
        messages = [sysMsg, HumanMessage(content=prog)]
        default_response = {
          'class':'None','type': 'None', 'subtype': 'None'
        }
        response = self.try_to_invoke(messages, default_response)
        try:
          print(f"{response['parsed']}")
          print(f"{response['raw']}")
        except:
          print(response)
          response = {'parsed':response, 'raw':response}
        try:
          reasoning = response['raw'].additional_kwargs.get("reasoning_content")
          #print(f"Reasoning:{reasoning}")
        except Exception as e:
          print(f"Cannot extract reasoning tokens: {e}")

        classification_data['classification'].append(response['parsed'])
        verified_prg+=1
        
        if save_usage_metadata:
          self.get_usage_metadata(response, verified_prg-1)

        print(f"Progress: {verified_prg}/{len(prg_paths)}")
        print("-"*20+" Sleep inserted to avoid consuming all tokens "+"-"*20)
        time.sleep(15)
    self.last_run_time = time.time()-start
    classification_data['id'] = [i for i in range(verified_prg)]
    
    if self.logging:
      log_tool_interactions(response)

    res = self.try_into_dataframe(classification_data)
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
        
  def try_into_dataframe(self, data):
    try:
      res = pd.DataFrame(data)
    except Exception as e:
      print(f"Error while transforming into dataframe: {e}")
      res = pd.DataFrame()
      with open("result_"+self.model+".json", "w") as f:
        f.write(json.dumps(data, indent=4))
    return res

  def get_usage_metadata(self, response, msg_id:int = 0):
    """The function tries to extract the usage_metadata from the 'response'.
    The couple '(msg_id, response.usage_metadata)' is appended in the usage_metadata field."""
    if isinstance(response, dict):
      try:
        raw = response['raw']
      except Exception as e:
        self.usage_metadata.append(f"Unable to fetch the response. {e}")
        print(response)
        return
    
    if isinstance(raw,list):
      for msg in raw:
        if isinstance(msg, AIMessage): 
          if hasattr(msg, 'usage_metadata'):
            self.usage_metadata.append((msg_id,msg.usage_metadata))
          elif hasattr(msg,'response_metadata'):
            self.usage_metadata.append((msg_id,msg.response_metadata))
          elif hasattr(msg,'additional_kwargs'):
            self.usage_metadata.append((msg_id,msg.additional_kwargs))
    
    elif isinstance(raw,dict):
      try:
        self.usage_metadata.append((msg_id, raw['response_metadata']))
      except Exception as e:
        print(f"Could not append metadata: {e}")
    elif isinstance(raw,AIMessage):
      try:
        self.usage_metadata.append((msg_id, raw.response_metadata))
      except Exception as e:
        print(f"Could not append metadata: {e}")
      
      
  def get_token_count(self):
    """The function tries to fetch the saved usage_metadata to extract the input and output tokens and returns them.
    In case the extraction fails it returns (-1,-1)."""
    if self.usage_metadata == []:
      return (-1,-1)
    input_tokens,output_tokens = 0,0
    for id,metadata in self.usage_metadata:
      try:
        usage = metadata.get("token_usage", {}) # Try generic name first
        if not usage:
            # Fallback to Ollama key names
            input_tokens += metadata.get("prompt_eval_count", 0)
            output_tokens += metadata.get("eval_count", 0)
        else:
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
      except Exception as e:
        print(f"Error during extraction of token count: {e}")
        return -1,-1
    return input_tokens,output_tokens
    
  def print_token_count(self):
    """The function prints the input, output and total tokens usage extracted from the usage_metadata.
    In case there is an error during the extraction it immediately returns, without printing anything."""
    input_tokens,output_tokens=self.get_token_count()
    if input_tokens == -1:
      return
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens: {input_tokens+output_tokens}")

    
#Groq models: qwen/qwen3-32b, llama-3.3-70b-versatile, llama-3.1-8b-instant, meta-llama/llama-4-maverick-17b-128e-instruct,
# moonshotai/kimi-k2-instruct-0905
if __name__=='__main__':
  a = VerificationAgent(
    provider='Groq',
    model='llama-3.1-8b-instant',
    test_paths='benchmarks_paths/test_set.txt', 
    validation_paths='benchmarks_paths/validation_set.txt'
    )
  df = a.run_on_benchmark(validation=True, save_usage_metadata=True)
  print(f"Time required: {a.last_run_time}")
  print(f"Usage metadata: {a.usage_metadata}")
  a.print_token_count()
  stop = False
  while not stop:
    try:
      df.to_csv(f"benchmark_results_{a.model}.csv", index=False)      
      stop = True
    except Exception as e:
      stop = False
      print(f"Exception: {e}")
      i = input("Retry? [Y/N]")
      if i != "Y":
        stop = True
    


