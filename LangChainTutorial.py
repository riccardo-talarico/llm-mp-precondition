import getpass
import os
from dotenv import load_dotenv

load_dotenv()

#if not os.environ.get("GEMINI_API_KEY"):
  #os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter API key for Gemini: ")


# Architecture-
# langchain-core: base abstraction for chat models
# langchain: Chains,agents and retrieval strategies
# langgraph: Orchestration framework for combining langchain components into a production-read app
# langsmith: platform to support evaluation and observability for AI-apps

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import FilesystemFileSearchMiddleware
#from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelRequestHandler
from langchain.agents.middleware.types import ModelResponse
from utils.tool_analysis import log_tool_interactions

#class LoggingMiddleware(AgentMiddleware):
#    def wrap_model_call(self, request: ModelRequest, handler: ModelRequestHandler) -> ModelResponse:
#        # Before the model runs
#        print("=== Before model call ===")
#        print("Messages:", request.state.messages)
#        resp = handler(request)
#        # After the model runs
#        print("=== After model call ===")
#        print("Model responded with:", resp)
#        return resp
#
#    def wrap_tool_call(self, tool_name: str, tool_input: any, handler):
#        # This is called when any tool is invoked
#        print(f"Calling tool `{tool_name}` with input: {tool_input}")
#        result = handler(tool_name, tool_input)
#        print(f"Result from tool `{tool_name}`: {result}")
#        return result

# successful tool call: {'name': 'grep_search', 'args': {'path': 'file-parser', 'pattern': '.', 'include': 'main.go', 'output_mode': 'content'}, 'id': 'fc8744a2-f665-4bc0-8d64-290a18fd0847', 'type': 'tool_call'}]


if __name__ == '__main__':
  api_key = os.getenv("GEMINI_API_KEY")
  #model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

  """Model key methods:
  invoke
  stream: like invoke, but stream the output as it is generated in real-time
  batch: send multiple requests to a model in a batch for more efficient processing
  """

  llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)
  
  """ message objects contain:
  - role: identify message type (system or user)
  - content
  - metadata: optional field (e.g. messageId, token_usage)
  from lanchain.messages import SystemMessage, HumanMessage, AIMessage
  Messages types:
  - Human
  - System: tells the model how to behave and provide context for interactions
  - AI
  - Tool: represents the output of a tool call
  """

  messages = [
      SystemMessage(content='The user will provide a folder name and you will search for the go file inside it.' +
      'use the file_search middleware to find the file'+
      "If you find the file here are the args to pass to get access to the content: 'path': [dir_name], 'pattern': '.', 'include': '[filename].go', 'output_mode': 'content'"
      '. Read the file and act as an expert verifier to provide '+
      'the weakest precondition that can ensure partial deadlock freedom ' +
      '(if you are not sure what the weakest precondition is, provide a general precondition, if you cannot even do this write UNKNOWN in the precondition field of the response and provide a concise explanation of why you dont have a precondition) '+
      'The weakest precondition must regard the concurrency parameters of the fragment you are analyzing and must be a boolean expression'+
      ' that can act as an assert condition. This is the structure of the response you must provide: '\
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
      #checkpointer = InMemorySaver()
      # state_schema = can pass CustomAgentState
      # response_state = can set desired structured output schema
      # type Union[ToolStrategy[StructuredResponseT],ProviderStrategy[StructuredResponseT],type[StructuredResponseT],]
    )
  
  response = agent.invoke(msg)
  print(f"Full response object: {response}")
  print("="*40)
  print(f"Response: {response['messages'][-1].content[0]['text']}")
  
  log_tool_interactions(response)
  print(f"Metadata: {response['messages'][-1].usage_metadata}")

