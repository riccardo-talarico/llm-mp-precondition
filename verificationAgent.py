import getpass
import os, time, json
from dotenv import load_dotenv

load_dotenv()

from google.api_core import exceptions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import SummarizationMiddleware, FilesystemFileSearchMiddleware
from langchain_core.runnables import RunnableConfig
from utils.tool_analysis import log_tool_interactions
import pandas as pd


class VerificationAgent():

  def __init__(self, benchmark_folder,provider='Google',model="gemini-2.5-flash", propose_fix=False, logging = True):
    if provider == 'Google':
      api_key = os.getenv("GEMINI_API_KEY")
      self.llm = ChatGoogleGenerativeAI(model=model,google_api_key=api_key, max_retries = 3)
    elif provider == 'Groq':
      api_key = os.getenv('GROQ_API_KEY')
      self.llm = ChatGroq(model=model, groq_api_key=api_key)
    else:
      print("Other providers not currently supported")
      self.llm = None  
    self.folder = benchmark_folder
    self.logging = logging
    self.model = model
    self.usage_metadata = []

    # Defining JSON response schema
    properties_schemas = {}
    properties_schemas['categorization'] = {
      'type': "object",
      'description':"Provide a categorization for the bug found in the code based on the GoBench paper classification",
      'properties': {
        'subtype':{'type':'string','description':'Subtype of the bug, chosen according to the GoBench paper'},
        'subsubtype':{'type':'string','description':'Subsubtype of the bug, chosen according to the GoBench paper'}
      },
      'required':['subtype','subsubtype']
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
      
  def run_on_benchmark(self, save_usage_metadata=True):
    projects = os.listdir(self.folder)

    sysMsg = SystemMessage(content='Role: You are a Go Concurrency Expert and Program Verifier. Your task is to analyze Go code snippets to identify and classify blocking bugs according to the following hierarchy.'+\
'Bug Classification Hierarchy: If a blocking bug is detected, classify it into exactly one subtype and subsubtype based on these definitions:'+
'1. Resource Deadlock: Goroutines block waiting for a synchronization resource (lock) held by another. Subsubtypes:'+
'1.1. Double Locking: A single goroutine attempts to acquire a lock it already holds, causing it to block itself.'+
'1.2. AB-BA Deadlock: Multiple goroutines acquire multiple locks in conflicting orders (e.g., G1: Lock A then B; G2: Lock B then A).'+
'1.3. RWR Deadlock: Involving sync.RWMutex. A pending Write lock request takes priority, blocking subsequent Read requests even if the current lock is a Read lock, potentially creating a cycle if the current Reader waits for a new Reader.'+
'2. Communication Deadlock: Goroutines block waiting for a message/signal from another. Subsubtypes:' +
'2.1. Channel: Sending/Receiving on a channel where no counterpart is available to complete the handoff (e.g., unbuffered channel leaks).' +
'2.2. Condition Variable: Misuse of sync.Cond (e.g., Wait() is called but Signal() or Broadcast() is never triggered due to logic errors).' +
'2.3. WaitGroup: Calling Wait() on a sync.WaitGroup where the internal counter never reaches zero due to missing Done() calls.'+
'2.4. Channel & Context: Complex communication blocks involving the interaction of channels with context.'+
'2.5. Channel & Condition Variable: Complex communication blocks involving the interaction of channels with condition variables.'+
'3. Mixed Deadlock: A cycle created by mixing message-passing and shared-memory synchronization. Subsubtypes:'+
'3.1. Channel & Lock: A cycle where a goroutine holds a lock while waiting for a channel operation, while the counterpart for that channel operation is waiting for the same lock.'+
'3.2. Channel & WaitGroup: A cycle where a channel operation is blocked by a WaitGroup.Wait(), or a WaitGroup.Done() is blocked by a channel operation.'+
'Verification Logic:'+
'Assume Partial Context: If the snippet is missing a main function, assume the provided functions are called in a way that triggers the concurrency logic shown.'+
'Strict Classification: Use only the specific subtype (Resource Deadlock, Communication Deadlock or Mixed Deadlock) and subsubtype name (e.g., Channel & Lock) in your response. DO NOT USE OTHER LABELS.'+
'If no bug is found then insert None both as the subtype and subsubtype. Follow the JSON structure provided'
      )
    classification_data = {'id':[], 'classification': []}
    usage_metadata = []
    verified_prg = 0
    for proj in projects:
      proj_folder = os.path.join(self.folder,proj)
      for fragment in os.listdir(proj_folder):
        frag_path = os.path.join(proj_folder, fragment)
        frag_path = os.path.join(frag_path, proj+fragment+"_test.go")

        with open(frag_path, "r") as f:
          print(f"File: {frag_path}")
          prog = f.read()
          messages = [sysMsg, HumanMessage(content=prog)]
          default_response = {
            'categorization': {'subsubtype': 'Skipped', 'subtype': 'Skipped'},
          }
          response = self.try_to_invoke(messages, default_response)
          print(f"{response['parsed']}")
          try:
            reasoning = response['raw'].additional_kwargs.get("reasoning_content")
            print(f"Reasoning:{reasoning}")
          except Exception as e:
            print(f"Cannot extract reasoning tokens: {e}")

          classification_data['classification'].append(response['parsed'])
          verified_prg+=1
          
          if save_usage_metadata:
            self.get_usage_metadata(response, verified_prg-1)

          print(f"Progress: {verified_prg}/68")
          print("-"*20+" Sleep inserted to avoid consuming all tokens "+"-"*20)
          time.sleep(15)
          
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

  def get_usage_metadata(self, response, msg_id = 0):

    if isinstance(response, dict):
      try:
        response = response['raw']
      except Exception as e:
        self.usage_metadata.append(f"Unable to fetch the response. {e}")
        return

    for msg in response:
      if isinstance(msg, AIMessage) and hasattr(msg, 'usage_metadata'):
        self.usage_metadata.append((msg_id,msg.usage_metadata))

    
#Groq models: qwen/qwen3-32b, llama-3.3-70b-versatile, llama-3.1-8b-instant, meta-llama/llama-4-maverick-17b-128e-instruct,
# moonshotai/kimi-k2-instruct-0905
if __name__=='__main__':
  a = VerificationAgent(provider='Groq',model='llama-3.1-8b-instant',benchmark_folder='gomela/benchmarks/blocking')
  df = a.run_on_benchmark()
  print(a.usage_metadata)
  print(df)
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
    


