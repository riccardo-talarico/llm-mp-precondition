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
      SystemMessage(content='The user will provide a folder name and you will search for the main file inside it.' +
      'use the file_search middleware to find the file'+
      '. Read the file and act as an expert verifier to provide '+
      'the weakest precondition that can ensure partial deadlock freedom ' +
      '(so the termination of all go routines) '
      ),
      HumanMessage(content='file-parser')
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
  print(f"Response: {response['messages'][-1].content}")


  print(f"Metadata: {response['messages'][-1].usage_metadata}")


# TODO: load a file containing some code and let the agent analyze it
# finish tutorial on semantic search with documents and then copy the code for creating
# an agent that can use search engines/ other tools (?) and query it to analyze the code