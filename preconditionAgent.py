import getpass
import os
from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import SummarizationMiddleware, FilesystemFileSearchMiddleware
from langchain_core.runnables import RunnableConfig
from utils.tool_analysis import log_tool_interactions
import pandas as pd


concurrency_params_schema = {
  'type': ['string', 'null'], 
  'description':'Concurrency parameters found in the code'
}
weakest_precond_schema = {
  'type': ['string', 'null'], 
  'description':'Weakest precondition to ensure partial deadlock freedom. Must be a boolean expression that can be used as an assert condition',
  'examples': ['x==0','len(list) > 10','False','True','send+y<=receive']
}



class PreconditionAgent():

  def __init__(self, benchmark_folder,provider='Google', propose_fix=False, logging = True):
    if provider == 'Google':
      api_key = os.getenv("GEMINI_API_KEY")
      self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)
    else:
      # TODO: add options for other provides
      self.llm = None  
    self.folder = benchmark_folder
    self.logging = logging
    self.usage_metadata = []

    # Defining JSON response schema
    self.response_schema = {
      "type": "object",
      "description": "response schema for preconditions found",
      "properties" : {
        'concurrency_parameters': concurrency_params_schema,
        'weakest_precondition': weakest_precond_schema,
      },
      'required':['concurrency_parameters','weakest_precondition']
    }
    if propose_fix:
      self.response_schema['properties']['proposed_fix']={
        'type': ['string','null'],
        'description':'Suggest eventual fixes to the program to avoid deadlock freedom. Must be shown as git diffs.',
        'examples': ['-ch:=make(chan) +ch:=make(chan,1)']
      }
      self.response_schema['required'].append('proposed_fix')
    
    # Creating tools and middlewares
    file_search = FilesystemFileSearchMiddleware(
      root_path="./" + self.folder,
      use_ripgrep=True  # Fast regex search
      )
    summary = SummarizationMiddleware(
      model=self.llm, 
      trigger=("tokens", 4000),
      keep=("messages", 20)
      )

    # Creating agent
    self.agent = create_agent(
      model = self.llm, 
      tools = [],
      middleware = [
        file_search,
        summary
      ],
      checkpointer = InMemorySaver(),
      response_format = ToolStrategy(self.response_schema)
    )
    
    self.config : RunnableConfig = {"configurable": {"thread_id": "1"}}
  
  def run_on_benchmark(self, save_usage_metadata=True):
    programs = os.listdir(self.folder)
    
    messages = [
      SystemMessage(content='The user will provide a folder name and you will search for the go file inside it.' +
      'use the file_search middleware to find the file'+
      "If you find the file, here are the args to pass to get access to the content: 'path': [dir_name], 'pattern': '.', 'include': '[filename].go', 'output_mode': 'content'."
      'Read the file and act as an expert verifier to provide '+
      'the weakest precondition that can ensure partial deadlock freedom ' +
      '(if you are not sure what the weakest precondition is, provide a general precondition, if you cannot even do this write UNKNOWN in the precondition field of the response and provide a concise explanation of why you dont have a precondition) '+
      'The weakest precondition must regard the concurrency parameters of the fragment you are analyzing and must be a boolean expression'+
      ' that can act as an assert condition. Examples: if the program fragment has no possible condition to ensure deadlock freedom put return False as precondition,' +
      ' if the program works only if the concurrency parameter is 0 or its length is equal to 0 than you can set: param==0 (or len(param)==0)'
      ),
    ]

    precondition_data = {'id':[i for i in range(len(programs))], 'precondition':[]}
    usage_metadata = []
    for dir in programs:
      messages.append(HumanMessage(content=dir))
      msg = {'messages': messages}

      response = self.agent.invoke(msg, self.config)
      precondition_data['precondition'].append(response['structured_response']['weakest_precondition'])
      print(f"{response['structured_response']}")
      
      messages = []

    if self.logging:
      log_tool_interactions(response)

    if save_usage_metadata:
        self.get_usage_metadata(response)

    return pd.DataFrame(precondition_data)
  
  def get_usage_metadata(self, response):
    # To keep track of the number of messages
    num_message = 0

    if isinstance(response, dict):
      try:
        response = response['messages']
      except KeyError:
        self.usage_metadata.append(f"Unable to fetch the response, Key error: messages")
        return

    for msg in response:
      if isinstance(msg, AIMessage) and hasattr(msg, 'usage_metadata'):
        self.usage_metadata.append((num_message,msg.usage_metadata))
        num_message+=1

    

if __name__=='__main__':
  a = PreconditionAgent(benchmark_folder='benchmarks')
  df = a.run_on_benchmark()
  print(a.usage_metadata)
  print(df)
  df.to_csv("benchmark_results.csv", index=False)


