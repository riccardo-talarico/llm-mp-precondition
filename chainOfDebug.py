from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
import os, time
import pandas as pd
import json
from dotenv import load_dotenv
from utils.prompts import *
from utils.graph import *
from utils.go_parsing import parse_go_concurrency
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
from utils.tool_analysis import log_tool_interactions
from langchain.agents.structured_output import ToolStrategy

load_dotenv()

#TODO: substitue node 1 with a parsing function (python or C) to detect all the concurrency primitives
# automatically. Idea: adding RAG to trace generations, so that it has possible examples.


class ChainOfDebugAgent():
    def __init__(self, provider : str, model : str, json_mode=False, debug_level: int = 0, logging = True):
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
        self.logging = logging
        
    def _get_concurrency_primitives(self, state : State, config : RunnableConfig, node_name : str = None):
        """The function describes the first node logic: 
        it runs a script to identify concurrency structures and functions using them"""        
        return {'concurrency_primitives': parse_go_concurrency(state['code'])}

    @handle_early_exit("trace_list")  
    def _generate_traces(self, state: State, config: RunnableConfig, node_name: str = None):
        """The function implements the logic of the generate_trace node:
        it asks the llm to generate a list of problematic traces, 
        given the concurrency primitives identified."""
        
        schema = Traces.model_json_schema()
        sysprompt = self.generate_list_of_traces_prompt.format(primitives=state["concurrency_primitives"])
        input = [SystemMessage(sysprompt), HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(Traces, method=self.structured_output_method, include_raw=True)
        msg = self.try_to_invoke(input, structured_llm, node_name)

        return msg
    
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
        """This function describes the logic of the guard node that checks if the agent found a bug"""
        
        if state['early_stop']:
            return "STOP"
        elif state["trace_eval"].reachable is False: 
            return "NO BUG"
        else:
            return "FOUND"
    
    @handle_early_exit("trace_eval")
    def _ask_if_trace_is_possible(self, state : State, config: RunnableConfig, node_name: str = None):
        """The function describes the logic of the check_trace node: 
        it asks the llm to verify if the trace is possible or if it originates from 
        an impossible execution path."""
        
        schema = TraceEvaluation.model_json_schema()
        input = [SystemMessage(self.verify_trace_prompt.format(trace=state["active_trace"])),HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(TraceEvaluation, method=self.structured_output_method, include_raw=True)
        msg = self.try_to_invoke(input,structured_llm, node_name)
        return msg


    @handle_early_exit("classification")
    def _create_classification(self, state : State, config: RunnableConfig, node_name: str = None):
        """The function describes the logic of the create classification node: 
        it asks the llm to classify the bug, given the program and the problematic trace."""

        classification_schema = BugClassification.model_json_schema()
        input = [SystemMessage(self.classification_prompt.format(trace=state['active_trace'], trace_eval=state["trace_eval"])), HumanMessage("Code:\n"+state["code"])]
        structured_llm = self.llm.with_structured_output(BugClassification, method = self.structured_output_method, include_raw=True)
        msg = self.try_to_invoke(input,structured_llm, node_name)
        return msg
    
    def _empty_classification(self, state: State):
        """Function that describe the logic of the empty classification node: 
        since no problematic trace was found, there is no bug to report."""
        return {"classification" : None}
    
    def _early_termination(self, state : State):
        """Routing function to handle early stopping"""
        return "STOP" if state["early_stop"] else "NO STOP"

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
        self.graph.add_conditional_edges(
            "get_concurrency_primitives", self._early_termination, {"STOP": END, "NO STOP": "generate_traces"})
        self.graph.add_conditional_edges(
            "generate_traces", self._early_termination, {"STOP": END, "NO STOP": "trace_selector"}
        )
        self.graph.add_conditional_edges(
            "trace_selector", self._check_if_trace_list_is_empty, {"EMPTY": "empty_classification", "NOT EMPTY": "check_trace"}
        )
        self.graph.add_conditional_edges(
            "check_trace", self._check_if_found_bug, {"NO BUG": "trace_selector", "FOUND": "create_classification", "STOP": END} 
            )
        self.graph.add_edge("empty_classification", END)
        self.graph.add_edge("create_classification", END) 
        
        self.graph = self.graph.compile()
        if self.graph != None:
            self.compiled = True
        if save_img:
            save_graph_img(self.graph, "chain_of_debug")    

    def invoke(self, code : str):
        response = self.graph.invoke({"code":code, "early_stop": False})
        if self.logging:
            log_tool_interactions(response)
        return response
    
    def try_to_invoke(self, msg, llm, node_name : str):
        """The function tries to invoke llm with input msg, in case the operation fails
        the user is asked to input Y/N to choose whether to abort the call graph or not"""
        while True:
            try:
                response = llm.invoke(msg)
                return response
            except Exception as e:
                print(f"Error in node: [{node_name}]:{e}")
                # cleanest way is to use the interrupt function from langgraph
                s = input("Abort graph call? [Y/N]")
                if s == 'Y':
                    return "ABORTED"


    def load_prompts(self):
        self.identify_concurrency_prompt = IDENTIFY_CONCURRENCY_PROMPT
        self.generate_list_of_traces_prompt = GENERATE_TRACES_PROMPT
        self.verify_trace_prompt = VERIFY_TRACE_PROMPT
        self.classification_prompt = CLASSIFICATION_PROMPT

    def run_on_benchmark(self, paths_file : str, save_usage_metadata=True):
        """The function run the chain agent on all the programs specified by 'paths_file'."""
        try:
            with open(paths_file, "r") as f:
                prg_paths = f.readlines()
        except Exception as e:
            print(f"Unable to read the paths_file: {e}")
            return 

        classification_data = {'id':[], 'classification': []}
        thinking_log = {'id':[], 'thinking':[]}
        #usage_metadata = []
        verified_prg = 0
        for prg_path in prg_paths:
            # [:-1] to ignore the '\n'
            with open(prg_path[:-1], "r") as f:
                print(f"File: {prg_path[:-1]}")
                prog = f.read()
                response = self.invoke(prog)
                classification_data['classification'].append(response['classification'])
                thinking_log['thinking'].append(response['reasoning'])
                print(f"{response['classification']}")
                verified_prg+=1

                #if save_usage_metadata:
                #    self.get_usage_metadata(response, verified_prg-1)

                print(f"Progress: {verified_prg}/{len(prg_paths)}")
                print("-"*20+" Sleep inserted to avoid consuming all tokens "+"-"*20)
                time.sleep(10)
            
        classification_data['id'] = [i for i in range(verified_prg)]
        thinking_log['id'] = [i for i in range(verified_prg)]
        
        res = self.try_into_dataframe(classification_data)
        try:
            with open("result_"+self.model+".json", "w") as f:
                f.write(json.dumps(thinking_log, indent=4))
        except Exception as e:
            print(f"Cannot save thinking data: {e}")

        return res
    
    def try_into_dataframe(self, data):
        """The function tries to save the data into a dataframe. 
        In case the operation fails, the data is saved into a json file."""
        try:
            res = pd.DataFrame(data)
        except Exception as e:
            print(f"Error while transforming into dataframe: {e}")
            res = pd.DataFrame()
            with open("result_"+self.model+".json", "w") as f:
                f.write(json.dumps(data, indent=4))
        return res



# llama-3.3-70b-versatile, qwen/qwen3-32b (No support tool calling), 
# moonshotai/kimi-k2-instruct, llama-3.1-8b-instant
# meta-llama/llama-4-scout-17b-16e-instruct, groq/compound-> no support for tool calling
# meta-llama/llama-4-maverick-17b-128e-instruct
if __name__ == '__main__':
    a = ChainOfDebugAgent(provider='Groq', model='llama-3.1-8b-instant', json_mode=False, debug_level=1)
    a.compile_chain(save_img=True)
    # Running the benchmark on the validation set
    df = a.run_on_benchmark("benchmarks_paths/validation_set.txt")
    df.to_csv(f"benchmark_results_{a.model}.csv", index=False)

