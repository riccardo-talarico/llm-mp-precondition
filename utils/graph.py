from langgraph.graph import StateGraph
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from functools import wraps
from typing_extensions import TypedDict, Literal, List, Annotated
from operator import add
from pydantic import BaseModel, Field, AliasChoices
from IPython.display import Image


class Trace(BaseModel):
    interleaving_logic : str = Field(
        validation_alias=AliasChoices("interleaving_logic","Logic","logic","Interleaving_logic"),
        description = "Concise explanation of the logic of the trace"
        )
    sequence : List[str] = Field(
        validation_alias=AliasChoices("sequence","Sequence"),
        description = "List of action step (strings that describe the action) that compose the trace"
        )
    @classmethod
    def get_json_template(cls):
        return """{"interleaving_logic": "string","sequence": [{"goroutine": "string", "action": "string"}]}"""

class Traces(BaseModel):
    traces: List[Trace] | None = Field(
        validation_alias=AliasChoices("traces","Traces","trace_list"),
        description = "list of objects of the class Trace"
        )
    @classmethod
    def get_json_template(cls):
        return """{"traces": [{"interleaving_logic": "string","sequence": [{"goroutine": "string", "action": "string"}]}]}"""

class TraceEvaluation(BaseModel):
    reachable: bool = Field(validation_alias=AliasChoices("reachable", "Reachable", "reachability"), description="Just a True or False value regarding the possibility of the program to create this trace")
    explanation: str = Field(validation_alias=AliasChoices("explanation", "Explanation", "motivation"))
    @classmethod
    def get_json_template(cls):
        return """{"reachable": true | false,"explanation": "string"}"""

class BugClassification(BaseModel):
    subtype : Literal["Resource Deadlock", "Communication Deadlock", "Mixed Deadlock"]
    subsubtype : Literal[
        "Double Locking", "AB-BA Deadlock", "RWR Deadlock",
        "Channel", "Condition Variable", "WaitGroup", "Channel & Context", "Channel & Condition Variable",
        "Channel & Lock", "Channel & WaitGroup"
        ]
    @classmethod
    def get_json_template(cls):
        return """{
    "subtype": "Resource Deadlock | Communication Deadlock | Mixed Deadlock",
    "subsubtype": "Double Locking | AB-BA Deadlock | RWR Deadlock | Channel | WaitGroup | Channel & Context | Channel & Condition Variable | Channel & Lock | Channel & WaitGroup"
}"""

class State(TypedDict):
    code : str
    early_stop: bool
    # Keeps a list of reasoning tokens, the field is not overwritten every time
    reasoning: Annotated[List[str], add]
    concurrency_primitives : str 
    classification : BugClassification | None
    trace_list : Traces
    active_trace : Trace | None 
    trace_eval : TraceEvaluation | None



def handle_early_exit(state_key : str):
    def decorator(func):
        @wraps(func)
        def early_exit(self, state, config : RunnableConfig = None, **kwargs):
            node_name = "unknown_node"
            if config and "metadata" in config:
                node_name = config["metadata"].get("langgraph_node", node_name)
            result = func(self, state, config, node_name=node_name, **kwargs)
            if isinstance(result,str) and result == "ABORTED":
                return {'early_stop': True}
            # Otherwise wrap the expected result in the corresponding key
            try:
                reasoning = result['raw'].additional_kwargs.get("reasoning_content")
            except Exception as e:
                print(f"Cannot extract reasoning tokens: {e}")
            return {state_key: result['parsed'], 'reasoning_log': [reasoning]}
        return early_exit
    return decorator

def save_graph_img(graph: StateGraph, name:str = "graph"):
    img = Image(graph.get_graph(xray=True).draw_mermaid_png())

    with open(f"{name}.png","wb") as f:
        f.write(img.data)

def get_universal_template(model_cls):
    """Generates a simple JSON skeleton from any Pydantic model."""
    return {field: "string" for field in model_cls.model_fields}

if __name__ == '__main__':
    print(TraceEvaluation.model_json_schema())