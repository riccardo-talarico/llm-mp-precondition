from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Literal, List
from pydantic import BaseModel, Field, AliasChoices
from IPython.display import Image

# TODO: add __str__ methods for all the structured output, to make it more readable for the LLM 
class GoPrimitive(BaseModel):
    name : str = Field(validation_alias=AliasChoices("name","Name"), descrpition="The identifier of the primitive.")
    type : str = Field(validation_alias = AliasChoices("type","Type"), description="The exact Go type or primitive name (e.g., sync.Mutex, chan).")
    function : str = Field(validation_alias = AliasChoices("function","Function"), description="The name of the function where the primitive is declared or used.")
    scope : str = Field(validation_alias = AliasChoices("scope","scope/context", "Scopre"), description="The code context (e.g., 'inside a for-loop', 'guarded by an if-statement').")
    #def __str__(self):

class GoPrimitives(BaseModel):
    primitives : List[GoPrimitive] | None = Field(validation_alias=AliasChoices("concurrency_primitives", "Primitives", "Concurrency_primitives", "primitives"))
    class Config:
        # This allows to use GoPrimitives(primitives=...) in the code
        populate_by_name = True

class ActionStep(BaseModel):
    goroutine : str = Field(validation_alias=AliasChoices("Routine","goroutine","Goroutine","go_routine"))
    action : str = Field(validation_alias = AliasChoices("action","Action"))

class Trace(BaseModel):
    interleaving_logic : str = Field(validation_alias=AliasChoices("Logic","logic","Interleaving_logic","interleaving_logic"))
    sequence : List[ActionStep] = Field(validation_alias=AliasChoices("sequence","Sequence"))

class TraceEvaluation(BaseModel):
    reachable: bool = Field(validation_alias=AliasChoices("reachable", "Reachable", "reachability"), description="Just a True or False value regarding the possibility of the program to create this trace")
    explanation: str = Field(validation_alias=AliasChoices("explanation", "Explanation", "motivation"))


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
    trace_eval : TraceEvaluation | None


def save_graph_img(graph: StateGraph, name:str = "graph"):
    img = Image(graph.get_graph(xray=True).draw_mermaid_png())

    with open(f"{name}.png","wb") as f:
        f.write(img.data)