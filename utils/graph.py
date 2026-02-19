from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Literal, List
from pydantic import BaseModel
from IPython.display import Image

# TODO: add __str__ methods for all the structured output, to make it more readable for the LLM 
class GoPrimitive(BaseModel):
    name : str
    type : str
    function : str
    scope : str

class GoPrimitives(BaseModel):
    primitives : List[GoPrimitive] | None

class ActionStep(BaseModel):
    goroutine : str
    action : str

class Trace(BaseModel):
    interleaving_logic : str 
    sequence : List[ActionStep]

class BugClassification(BaseModel):
    subtype : Literal["Resource Deadlock", "Communication Deadlock", "Mixed Deadlock"]
    subsubtype : Literal[
        "Double Locking", "AB-BA Deadlock", "RWR Deadlock",
        "Channel", "Condition Variable", "WaitGroup", "Channel & Context", "Channel & Condition Variable",
        "Channel & Lock", "Channel & WaitGroup"
        ]

class State(TypedDict):
    code : str
    concurrency_primitives : GoPrimitives 
    classification : BugClassification | None
    trace : Trace | None 


def save_graph_img(graph: StateGraph, name:str = "graph"):
    img = Image(graph.get_graph(xray=True).draw_mermaid_png())

    with open(f"{name}.png","wb") as f:
        f.write(img.data)