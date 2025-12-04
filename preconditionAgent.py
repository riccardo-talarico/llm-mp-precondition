import getpass
import os
from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import SummarizationMiddleware, FilesystemFileSearchMiddleware
from utils.tool_analysis import log_tool_interactions


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

  def __init__(self, benchmark_folder,provider='Google', propose_fix=False):
    if provider == 'Google':
      api_key = os.getenv("GEMINI_API_KEY")
      self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)
    else:
      # TODO: add options for other provides
      self.llm = None  
    self.folder = benchmark_folder

    # initialize agent with tools
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
        # TODO: write more/better examples
        'examples': ['--ch:=make(chan) ++ch:=make(chan,1)']
      }
    file_search = FilesystemFileSearchMiddleware(
    root_path="./" + self.folder,
    use_ripgrep=True  # Fast regex search
    )
    # summarization middleware to summarize messages
    self.agent = create_agent(
      model = self.llm, 
      tools = [],
      middleware = [
        SummarizationMiddleware(
          model = llm,
          max_tokens_before_summary = 4000,
          messages_to_keep = 20
        ), file_search
      ],
      checkpointer = InMemorySaver(),
      response_format = ToolStrategy(self.response_schema)
    )
  
  def run_on_benchmark(self):
    messages = [
      SystemMessage(content='The user will provide a folder name and you will search for the go file inside it.' +
      'use the file_search middleware to find the file'+
      "If you find the file here are the args to pass to get access to the content: 'path': [dir_name], 'pattern': '.', 'include': '[filename].go', 'output_mode': 'content'"
      '. Read the file and act as an expert verifier to provide '+
      'the weakest precondition that can ensure partial deadlock freedom ' +
      '(if you are not sure what the weakest precondition is, provide a general precondition, if you cannot even do this write UNKNOWN in the precondition field of the response and provide a concise explanation of why you dont have a precondition) '+
      'The weakest precondition must regard the concurrency parameters of the fragment you are analyzing and must be a boolean expression'+
      ' that can act as an assert condition. Examples: if the program fragment has no possible condition to ensure deadlock freedom put return False as precondition,' +
      ' if the program works only if the concurrency parameter is 0 or its length is equal to 0 than you can set: param==0 (or len(param)==0)'
      ),
      HumanMessage(content='Ignore this message, in the next ones Ill provide the folder names' )
    ]
    for dir in os.listdir(self.folder):
      msg = {'messages': HumanMessage(content=dir)}
      response = self.agent.invoke(msg, {"configurable": {"thread_id": "1"}})
      #TODO: save responses somewhere 
      print(response['structured_response'])




if __name__ == '__main__':
  api_key = os.getenv("GEMINI_API_KEY")

  llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)

  messages = [
      SystemMessage(content='The user will provide a folder name and you will search for the go file inside it.' +
      'use the file_search middleware to find the file'+
      "If you find the file here are the args to pass to get access to the content: 'path': [dir_name], 'pattern': '.', 'include': '[filename].go', 'output_mode': 'content'"
      '. Read the file and act as an expert verifier to provide '+
      'the weakest precondition that can ensure partial deadlock freedom ' +
      '(if you are not sure what the weakest precondition is, provide a general precondition, if you cannot even do this write UNKNOWN in the precondition field of the response and provide a concise explanation of why you dont have a precondition) '+
      'The weakest precondition must regard the concurrency parameters of the fragment you are analyzing and must be a boolean expression'+
      ' that can act as an assert condition. Examples: if the program fragment has no possible condition to ensure deadlock freedom put return False as precondition,' +
      ' if the program works only if the concurrency parameter is 0 or its length is equal to 0 than you can set: param==0 (or len(param)==0)'
      ),
      HumanMessage(content='Respond to this message with a dot and wait for the folder names.')
  ]

  file_search = FilesystemFileSearchMiddleware(
    root_path="./benchmarks",
    use_ripgrep=True  # Fast regex search
  )

  precondition_schema = {
    "type": "object",
    "description": "response schema for preconditions found",
    "properties" : {
      'concurrency_parameters': {
        'type': ['string', 'null'], 
        'description':'Concurrency paramters found in the code'
        },
      'weakest_precondition': {
        'type': ['string', 'null'], 
        'description':'Weakest precondition to ensure partial deadlock freedom. Must be a boolean expression that can be used as an assert condition',
        'examples': ['x==0','len(list) > 10','False','True','send+y<=receive']
        }
    },
    'required':['concurrency_parameters','weakest_precondition']
  }
 
  # summarization middleware to summarize messages
  agent = create_agent(
      model = llm, 
      tools = [],
      middleware = [
        SummarizationMiddleware(
          model = llm,
          max_tokens_before_summary = 4000,
          messages_to_keep = 20
        ), file_search
      ],
      checkpointer = InMemorySaver(),
      response_format = ToolStrategy(precondition_schema)
    )
  msg = {'messages':messages}
  response = agent.invoke(msg, {"configurable": {"thread_id": "1"}})

  first = True
  for dir in os.listdir(path="./benchmarks"):
    #debug
    print(dir)
    dir_msg = HumanMessage(content=dir)
    msg = {'messages': dir_msg}
    response = agent.invoke(msg, {"configurable": {"thread_id": "1"}})
    
    print("="*60)
    print(f"Response: {response['structured_response']}")
    log_tool_interactions(response)
    print(f"Metadata: {response['messages'][-1].usage_metadata}")
    print("="*60)
    break

