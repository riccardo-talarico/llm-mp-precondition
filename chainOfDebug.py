from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from utils.prompts import *
from utils.graph import *
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

class ChainOfDebugAgent():
    def __init__(self, provider : str, model : str, json_mode=False, debug_level: int = 0):
        if provider == 'Google':
            self.llm = ChatGoogleGenerativeAI(model=model,api_key=os.getenv("GEMINI_API_KEY"))
        elif provider == 'Groq':
            api_key = os.getenv('GROQ_API_KEY')
            self.llm = ChatGroq(model=model, groq_api_key=api_key)
        else:
            self.llm = None
        self.graph = StateGraph(State)
        self.compiled = False
        self.structured_output_method = "json_mode" if json_mode else "function_calling" 
        self.load_prompts()
        self.debug_level = debug_level
        
    # 1. identify concurrency primitives and where they are used
    # 2. what is a possible trace that could cause problems with the given concurrency structure
    # 3. is this trace possible?
    # 4. classify the bug

    def _get_concurrency_primitives(self, state : State):
        """First call to identify concurrency structures and functions using them"""

        structured_llm = self.llm.with_structured_output(GoPrimitives, method=self.structured_output_method).with_retry(stop_after_attempt=1)
        
        sys_prompt = SystemMessage(self.identify_concurrency_prompt)
        prog_prompt = HumanMessage(f"Code:\n {state['code']}")
        input = [sys_prompt, prog_prompt]
        msg = structured_llm.invoke(input)
        return {"concurrency_primitives": msg}
    

    def _identify_trace(self, state: State):
        """Second call to generate a problematic trace given the concurrency primitives identified"""
        input = [SystemMessage(self.generate_trace_prompt.format(primitives=state["concurrency_primitives"])), HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(Trace, method=self.structured_output_method)
        msg = structured_llm.invoke(input)
        return {"active_trace" : msg}
    
    def _generate_traces(self, state: State):
        """Second call to generate a list of problematic traces, given the concurrency primitives identified"""
        input = [SystemMessage(self.generate_list_of_traces_prompt.format(primitives=state["concurrency_primitives"])), HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(Traces, method=self.structured_output_method)
        msg = structured_llm.invoke(input)

        if self.debug_level > 0:
            print(f"Traces produced: {len(msg.traces)}")
        return {"trace_list": msg}
    
    def _trace_selector(self, state:State):
        list = state["trace_list"]
        if len(list.traces)==0:
            return {"active_trace":None}
        else:
            if self.debug_level > 0:
                print(f"Remaining traces: {len(list.traces)}")
            active_trace = list.traces[0]
            list.traces = list.traces[1:]
            return {
                "active_trace": active_trace, 
                "trace_list": list 
                }
        
    def _check_if_trace_list_is_empty(self, state:State):
        """Guard node to check if the trace list is empty"""
        if state["active_trace"] is None:
            return "EMPTY"
        else:
            return "NOT EMPTY"

    def _check_if_found_bug(self, state : State):
        """Guard node to check if the agent found a bug"""
        
        # TODO: check logic for this
        if state["trace_eval"].reachable is False:
            return "NO BUG"
        else:
            return "FOUND"
            
    def _ask_if_trace_is_possible(self, state : State):
        """Asking the llm to verify if the trace is possible or if it originates from an impossible execution path"""
        input = [SystemMessage(self.verify_trace_prompt.format(trace=state["active_trace"])),HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(TraceEvaluation, method=self.structured_output_method)
        msg = structured_llm.invoke(input)
        return {"trace_eval":msg}

    def _create_classification(self, state : State):
        """Ask the llm to classify the bug, given the program and the problematic trace"""

        input = [SystemMessage(self.classification_prompt.format(trace=state['active_trace'], trace_eval=state["trace_eval"])), HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(BugClassification, method = self.structured_output_method)
        msg = structured_llm.invoke(input)
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
        self.graph.add_node("generate_traces",self._generate_traces)
        self.graph.add_node("trace_selector", self._trace_selector)
        self.graph.add_node("check_trace",self._ask_if_trace_is_possible)
        self.graph.add_node("create_classification", self._create_classification)
        self.graph.add_node("empty_classification", self._empty_classification)

        # Connecting edges
        self.graph.add_edge(START, "get_concurrency_primitives")
        self.graph.add_edge("get_concurrency_primitives", "generate_traces")
        self.graph.add_edge("generate_traces","trace_selector")
        self.graph.add_conditional_edges(
            "trace_selector", self._check_if_trace_list_is_empty, {"EMPTY": "empty_classification", "NOT EMPTY": "check_trace"}
        )
        self.graph.add_conditional_edges(
            "check_trace", self._check_if_found_bug, {"NO BUG": "trace_selector", "FOUND": "create_classification"} 
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

        #TODO: should I keep both prompts or just this one?
        self.generate_list_of_traces_prompt = GENERATE_TRACES_PROMPT
        self.generate_trace_prompt = GENERATE_TRACE_PROMPT
        self.verify_trace_prompt = VERIFY_TRACE_PROMPT
        self.classification_prompt = CLASSIFICATION_PROMPT
        
        if self.structured_output_method == "json_mode":
            add = "\nOutput must be a valid JSON object. Use the exact key names provided by the schema."
            schema = "Schema: {schema}"
            self.identify_concurrency_prompt += add +schema.format(schema="{primitives: list[GoPrimitive]}") +"\GoPrimitive Schema: {schema}".format(schema = "{name: , type: , function: , scope: }")
            self.generate_trace_prompt += add
            self.generate_list_of_traces_prompt += add
            self.verify_trace_prompt += add
            self.classification_prompt += add

# llama-3.3-70b-versatile, qwen/qwen3-32b
if __name__ == '__main__':
    a = ChainOfDebugAgent(provider='Groq', model='llama-3.3-70b-versatile', json_mode=False, debug_level=1)
    a.compile_chain(save_img=True)
    with open("gomela/benchmarks/blocking/cockroach/1462/cockroach1462_test.go", "r") as f:
        prg = f.read()
    res1 = a.invoke(prg)
    print(f"Reponse: {res1}")
    with open("./tool.log", "w") as f:
        f.write(str(res1))
    #import time
    #with open("gomela/benchmarks/blocking/cockroach/584/cockroach584_test.go", "r") as f:
    #    prg = f.read()
    #res1 = a.invoke(prg)
    #print(f"Reponse: {res1}")
    #time.sleep(30)
    #print("="*70)
    #with open("gomela/benchmarks/blocking/cockroach/16167/cockroach16167_test.go", "r") as f:
    #    prg = f.read()
    #res2 = a.invoke(prg)
    #print(f"Reponse: {res2}")
    #time.sleep(30)
    #print("="*70)
    #with open("gomela/benchmarks/blocking/cockroach/3710/cockroach3710_test.go", "r") as f:
    #    prg = f.read()
    #res3 = a.invoke(prg)
    #print(f"Reponse: {res3}")
#
    #with open("./tool.log", "w") as f:
    #    f.write(str(res1))
    #    f.write("\n")
    #    f.write(str(res2))
    #    f.write("\n")
    #    f.write(str(res3))
