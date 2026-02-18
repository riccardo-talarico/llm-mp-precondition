from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Literal, List
from pydantic import BaseModel, Field
from IPython.display import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from utils.prompts import *
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()


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



class ChainOfDebugAgent():
    def __init__(self, provider : str, model : str):
        if provider == 'Google':
            self.llm = ChatGoogleGenerativeAI(model=model,api_key=os.getenv("GEMINI_API_KEY"))
        elif provider == 'Groq':
            api_key = os.getenv('GROQ_API_KEY')
            self.llm = ChatGroq(model=model, groq_api_key=api_key)
        else:
            self.llm = None
        self.graph = StateGraph(State)
        self.compiled = False
        self.load_prompts()
    
    
    # 1. identify concurrency primitives and where they are used
    # 2. what is a possible trace that could cause problems with the given concurrency structure
    # 3. is this trace possible?
    # 4. classify the bug

    def _get_concurrency_primitives(self, state : State):
        """First call to identify concurrency structures and functions using them"""

        structured_llm = self.llm.with_structured_output(GoPrimitives).with_retry(stop_after_attempt=1)

        sys_prompt = SystemMessage(self.identify_concurrency_prompt)
        prog_prompt = HumanMessage(f"Code:\n {state['code']}")
        input = [sys_prompt, prog_prompt]

        msg = structured_llm.invoke(input)
        return {"concurrency_primitives": msg}
    

    #TODO: make it generate multiple traces, use SEND and verify them all
    # With send all the dynamic nodes have the same name, then you can just create an edge
    # between this name and the synthesizer node
    def _identify_trace(self, state: State):
        """Second call to generate a problematic trace given the concurrency primitives identified"""
        
        input = [SystemMessage(self.generate_trace_prompt.format(primitives=state["concurrency_primitives"])), HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(Trace)
        msg = structured_llm.invoke(input)
        return {"trace" : msg}

    def _check_if_found_bug(self, state : State):
        """Guard node to check if the agent found a bug"""
        
        if state["trace"] is None:
            return "NO BUG"
        else:
            return "FOUND"
            
    def _ask_if_trace_is_possible(self, state : State):
        """Asking the llm to verify if the trace is possible or if it originates from an impossible execution path"""

        input = [SystemMessage(self.verify_trace_prompt.format(trace=state["trace"])),HumanMessage("Code:\n"+state["code"])]
        msg = self.llm.invoke(input)
        return {"trace":msg}

    def _create_classification(self, state : State):
        """Ask the llm to classify the bug, given the program and the problematic trace"""

        input = [SystemMessage(self.classification_prompt.format(trace=state['trace'])), HumanMessage("Code:\n"+state["code"])]
        msg = self.llm.invoke(input)
        return {"classification": msg}
    
    def _empty_classification(self, state: State):
        """Since no problematic trace was found, the classification is empty"""

        return {"classification" : None}

    def compile_chain(self, save_img=False):
        if self.compiled:
            print("WARNING: you are recompiling an already compiled graph. It will be reinitialized")
            self.graph = StateGraph(State)
        
        # Adding the nodes in the graph
        self.graph.add_node("get_concurrency_primitives",self._get_concurrency_primitives)
        self.graph.add_node("generate_trace",self._identify_trace)
        self.graph.add_node("check_trace",self._ask_if_trace_is_possible)
        self.graph.add_node("create_classification", self._create_classification)
        self.graph.add_node("empty_classification", self._empty_classification)

        # Connecting edges
        self.graph.add_edge(START, "get_concurrency_primitives")
        self.graph.add_edge("get_concurrency_primitives", "generate_trace")
        self.graph.add_edge("generate_trace","check_trace")
        self.graph.add_conditional_edges(
            "check_trace", self._check_if_found_bug, {"NO BUG": "empty_classification", "FOUND": "create_classification"} 
            )
        self.graph.add_edge("empty_classification", END)
        self.graph.add_edge("create_classification", END) 
        
        self.graph = self.graph.compile()
        if self.graph != None:
            self.compiled = True
        if save_img:
            save_graph_img(self.graph, "chain_of_debug")    

    def invoke(self, code : str):
        response = self.graph.invoke({"code":code})
        return response


    def load_prompts(self):
        self.identify_concurrency_prompt = IDENTIFY_CONCURRENCY_PROMPT
        self.generate_trace_prompt = GENERATE_TRACE_PROMPT
        self.verify_trace_prompt = VERIFY_TRACE_PROMPT
        self.classification_prompt = CLASSIFICATION_PROMPT



def save_graph_img(graph: StateGraph, name:str = "graph"):
    img = Image(graph.get_graph(xray=True).draw_mermaid_png())

    with open(f"{name}.png","wb") as f:
        f.write(img.data)

if __name__ == '__main__':
    a = ChainOfDebugAgent(provider='Google', model='gemini-2.5-flash')
    a.compile_chain(save_img=False)
    
    with open("gomela/benchmarks/blocking/cockroach/584/cockroach584_test.go", "r") as f:
        prg = f.read()
    
    print(f"Reponse: {a.invoke(prg)}")