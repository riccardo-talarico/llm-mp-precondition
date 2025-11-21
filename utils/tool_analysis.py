from langchain.messages import AIMessage, ToolMessage
import logging

# Complet logging configuration, including date
logging.basicConfig(
    level=logging.DEBUG,
    filename="tool.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

def log_tool_interactions(response):
    interactions = get_tools_interactions(response)
    for id in interactions:
        logging.debug(f"Tool interaction: {interactions[id]}")


def get_tools_interactions(response):
    interactions = {}

    if isinstance(response, dict):
        try:
            response = response['messages']
        except KeyError:
            logging.error(f"Unable to fetch the response, Key error: messages", exc_info=True)
            return None
    
    for msg in response:
        if isinstance(msg, AIMessage):

            if 'function_call' in msg.additional_kwargs.keys():
                logging.debug(f"Function call to {msg.additional_kwargs['function_call']}")
            
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    interactions[tool_call['id']] = {
                        'name': tool_call['name'],
                        'args': tool_call['args'],
                        'response': 'Not Found'
                        }

        elif isinstance(msg, ToolMessage):
            call_id = msg.tool_call_id
            if call_id in interactions.keys():
                interactions[call_id]['response'] = msg.content
    
    return interactions



            
