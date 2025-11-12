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

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI


if __name__ == '__main__':
    api_key = os.getenv("GEMINI_API_KEY")
    #model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=api_key)
    print("Si")


# TODO: load a file containing some code and let the agent analyze it
# finish tutorial on semantic search with documents and then copy the code for creating
# an agent that can use search engines/ other tools (?) and query it to analyze the code