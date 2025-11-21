import getpass
import os
from dotenv import load_dotenv

load_dotenv()

#if not os.environ.get("GEMINI_API_KEY"):
  #os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter API key for Gemini: ")



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from utils.tool_analysis import log_tool_interactions

# successful tool call: {'name': 'grep_search', 'args': {'path': 'file-parser', 'pattern': '.', 'include': 'main.go', 'output_mode': 'content'}, 'id': 'fc8744a2-f665-4bc0-8d64-290a18fd0847', 'type': 'tool_call'}]


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
      ' if the program works only if the concurrency parameter is 0 or its length is equal to 0 than you can set: param==0 (or len(param)==0)'+  
      ' This is the structure of the response you must provide: '\
      ' Concurrency parameters: [list of concurrency params]'\
      ' Weakest Precondition: precondition'
      ),
      HumanMessage(content='negative-counter')
  ]

  msg = {
    'messages': messages
  }
  file_search = FilesystemFileSearchMiddleware(
    root_path="./benchmarks",
    use_ripgrep=True  # Fast regex search
  )

  # in production should use a memory backed by a database
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
      checkpointer = InMemorySaver()
      # state_schema = can pass CustomAgentState
      # response_state = can set desired structured output schema
      # type Union[ToolStrategy[StructuredResponseT],ProviderStrategy[StructuredResponseT],type[StructuredResponseT],]
    )
  
  response = agent.invoke(msg, {"configurable": {"thread_id": "1"}})
  print(f"Full response object: {response}")
  print("="*60)
  print(f"Response: {response['messages'][-1].content[0]['text']}")
  
  log_tool_interactions(response)
  print(f"Metadata: {response['messages'][-1].usage_metadata}")

