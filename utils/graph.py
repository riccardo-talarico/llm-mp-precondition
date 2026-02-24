from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Literal, List
from pydantic import BaseModel, Field, AliasChoices
from IPython.display import Image

# TODO: add __str__ methods for all the structured output, to make it more readable for the LLM 
class GoPrimitive(BaseModel):
    name : str = Field(validation_alias=AliasChoices("name","Name"), description="The identifier of the primitive.")
    type : str = Field(validation_alias = AliasChoices("type","Type"), description="The exact Go type or primitive name (e.g., sync.Mutex, chan).")
    function : str = Field(validation_alias = AliasChoices("function","Function"), description="The name of the function where the primitive is declared or used.")
    scope : str = Field(validation_alias = AliasChoices("scope","scope/context", "Scope"), description="The code context (e.g., 'inside a for-loop', 'guarded by an if-statement').")
    
    @classmethod
    def get_json_template(cls):
        json_template = """{"name": "string","type": "string","function":"string","scope":"string"}"""
        return json_template


class GoPrimitives(BaseModel):
    primitives : List[GoPrimitive] | None = Field(validation_alias=AliasChoices("primitives","concurrency_primitives", "Primitives", "Concurrency_primitives"))
    @classmethod
    def get_json_template(cls):
        return """{"primitives": [{"name": "string", "type": "string", "function": "string", "scope": "string"}]}"""

class ActionStep(BaseModel):
    goroutine : str = Field(validation_alias=AliasChoices("goroutine","Routine","Goroutine","go_routine"))
    action : str = Field(validation_alias = AliasChoices("action","Action"))
    @classmethod
    def get_json_template(cls):
        return """{"goroutine": "string","action": "string"}"""

class Trace(BaseModel):
    interleaving_logic : str = Field(
        validation_alias=AliasChoices("interleaving_logic","Logic","logic","Interleaving_logic"),
        description = "Concise explanation of the logic of the trace"
        )
    sequence : List[ActionStep] = Field(
        validation_alias=AliasChoices("sequence","Sequence"),
        description = "List of ActionStep that compose the trace"
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
    concurrency_primitives : GoPrimitives 
    classification : BugClassification | None
    trace_list : Traces
    active_trace : Trace | None 
    trace_eval : TraceEvaluation | None


def save_graph_img(graph: StateGraph, name:str = "graph"):
    img = Image(graph.get_graph(xray=True).draw_mermaid_png())

    with open(f"{name}.png","wb") as f:
        f.write(img.data)

def get_universal_template(model_cls):
    """Generates a simple JSON skeleton from any Pydantic model."""
    return {field: "string" for field in model_cls.model_fields}

if __name__ == '__main__':
    print(GoPrimitive.get_json_template())