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
        return """{"interleaving_logic": "string","sequence": ["string"]}"""


class TraceEvaluation(BaseModel):
    reachable: bool = Field(validation_alias=AliasChoices("reachable", "Reachable", "reachability"), description="Just a true or false value regarding the possibility of the program to create this trace")
    explanation: str = Field(validation_alias=AliasChoices("explanation", "Explanation", "motivation"))
    @classmethod
    def get_json_template(cls):
        return """{"reachable": "true | false","explanation": "string"}"""

class BugClassification(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of the trace and code to identify the bug type.")
    cls : Literal['Blocking', 'Nonblocking', 'None']
    type : Literal[
        # blocking:
        "Resource Deadlock", "Communication Deadlock", "Mixed Deadlock",
        # nonblocking:
        "Traditional", "Go-Specific",
        'None'
        ]
    subtype : Literal[
        # blocking:
        "Double locking", "AB-BA deadlock", "RWR deadlock",
        "Channel", "Condition Variable", "Misuse WaitGroup", "Channel & Context", "Channel & Condition Variable",
        "Channel & Lock", "Channel & waitGroup",
        # nonblocking:
        "Order violation", "Data race", "Misuse channel", "Anonymous function", "Testing library",
        'None'
        ]
    def get_classification(self):
        return {'cls':self.cls, 'type':self.type, 'subtype':self.subtype}
    @classmethod
    def get_json_template(cls):
        return """{
    "reasoning": "string",
    "cls": "Blocking | Nonblocking",
    "type": "Resource Deadlock | Communication Deadlock | Mixed Deadlock | Traditional | Go-specific",
    "subtype": "Double Locking | AB-BA Deadlock | RWR Deadlock | Channel | WaitGroup | Channel & Context | Channel & Condition Variable | Channel & Lock | Channel & WaitGroup
    | Order violation | Data race | Misuse channel | Anonymous function | Testing library"
}"""

class ConcurrencySection(BaseModel):
    primitives : List[str] = Field(
        description = "List of primitives involved in the concurrency section" 
    )
    scope : str = Field(
        description = "Brief description of the commands/code location composing the section"
    )
    @classmethod
    def get_json_template(cls):
        return """{"primitives":["string"],"scope":"string"}"""
    

# structured output for identify_sections node
class ConcurrencySections(BaseModel):
    sections: List[ConcurrencySection] = Field(
        description = "Concurrency sections found in the code."
    )
    @classmethod
    def get_json_template(cls):
        return """{
        "sections": {"primitives":["string"],"scope":"string"}
        }
"""

class SectionExplanation(BaseModel):
    section_objective : str = Field(
        description = "The goal of the section; what it is being used for."
    )
    section_functioning : str = Field(
        description = "Invariants that belong to the section and/or a lifecycle description."
    )
    @classmethod
    def get_json_template(cls):
        return """{
        "section_objective":"string", "section_functioning":"string"
        }"""

class BalanceReport(BaseModel):
    has_issues: bool = Field(
        description="True if any imbalance or potential issue is detected"
    )
    problems: list[str] = Field(
        description="List of concise problem descriptions. Empty if none."
    )
    summary: str = Field(
        description="Short overall assessment (max 25 words)"
    )
    @classmethod
    def get_json_template(cls):
        return """{
        "has_issues":"true | false", "problems":["string"], "summary":"string"
        }"""

class BugIdea(BaseModel):
    type: str = Field(
        description="Type of issue (e.g., deadlock, race_condition, goroutine_leak, waitgroup_misuse)"
    )
    description: str = Field(
        description="Short explanation of the potential issue (max 20 words)"
    )
    location: str = Field(
        description="Reference to the relevant section or code region"
    )
    support_info: list[str] = Field(
        description="Up to 2 short facts extracted from the analysis provided in the prompt supporting the idea"
    )
    @classmethod
    def get_json_template(cls):
        return """{
        "type":"string","description":"string","location":"string","support_info":"string"
        }
        """

class TraceIdeas(BaseModel):
    ideas : List[BugIdea] = Field(
        description = "List of possible bugs idea that will be developed into problematic traces"
    )
    @classmethod
    def get_json_template(cls):
        return """{
        "ideas":[{"type":"string","description":"string","location":"string","support_info":"string"}]
        }"""


class WorkerState(TypedDict):
    code : str
    section : ConcurrencySection
    section_explanations : Annotated[List[SectionExplanation], add]
    reasoning: Annotated[List[str], add]
    early_stop : bool

class TraceCreatorState(TypedDict):
    code : str
    trace_idea : BugIdea
    trace_list : Annotated[List[Trace], add]
    reasoning: Annotated[List[str], add]
    early_stop : bool
    #TODO: insert here classification?


class State(TypedDict):
    code : str
    early_stop: bool
    # Keeps a list of reasoning tokens, the field is not overwritten every time
    reasoning: Annotated[List[str], add]
    concurrency_primitives : str 
    concurrency_sections : ConcurrencySections
    section_explanations: Annotated[List[SectionExplanation], add]
    balance_report : BalanceReport
    trace_ideas : TraceIdeas | None
    classification : BugClassification | None
    trace_list : Annotated[List[Trace], add]
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
                return {'early_stop': True, 'classification':{'type':None,'subtype':None,'cls':None}}
            # Otherwise wrap the expected result in the corresponding key
            try:
                reasoning = [result['raw'].additional_kwargs.get("reasoning_content")]
            except Exception as e:
                print(f"Cannot extract reasoning tokens: {e}")
                reasoning = []

            try:
                state_update = result['parsed']
            except Exception as e:
                print(f"Exception during extraction of state update: {e}.\n Aborting the graph call")
                return {'early_stop':True, 'classification':{'type':None,'subtype':None,'cls':None}}

            return {state_key: state_update, 'reasoning': reasoning}
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
    print(ConcurrencySections.model_json_schema())