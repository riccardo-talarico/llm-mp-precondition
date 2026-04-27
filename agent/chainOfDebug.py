from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Overwrite
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

import os, time, json
from dotenv import load_dotenv

from utils.prompts import *
from utils.graph import *
from utils.output_parser import try_to_invoke
from utils.go_parsing import parse_go_concurrency
from utils.experiments import OllamaExperimentConfig, get_prompt_version, ExperimentLogger, load_prompt_string, extract_latest_prompt_version
from utils.tool_analysis import log_tool_interactions
from utils.results import extract_id, try_into_dataframe, get_usage_metadata, print_token_count

load_dotenv()

class ChainOfDebugAgent():
    def __init__(
            self, 
            provider : str, 
            model : str, 
            struct_output_method='json_schema', 
            debug_level: int = 0, 
            logging = True,
            ollama_cfg : None | OllamaExperimentConfig = None,
            prompt_config : dict | None = None,
            max_retries : int = 1,
            insert_sleep: bool = False
        ):
        if provider == 'Google':
            self.llm = ChatGoogleGenerativeAI(model=model,api_key=os.getenv("GEMINI_API_KEY"))
        elif provider == 'Groq':
            api_key = os.getenv('GROQ_API_KEY')
            self.llm = ChatGroq(model=model, groq_api_key=api_key, max_retries = max_retries)
        elif provider == 'Ollama':
            self.llm = ChatOllama(
            base_url=ollama_cfg.base_url,
            model=ollama_cfg.model,
            temperature=ollama_cfg.temperature,
            num_ctx=ollama_cfg.options.get("num_ctx", 4096),
            top_p=ollama_cfg.options.get("top_p", 0.9),
            top_k=ollama_cfg.options.get("top_k", 40),
            seed = ollama_cfg.seed,
            )
            prompt_config = ollama_cfg.prompt_config
            self.cfg = ollama_cfg
        else:
            self.llm = None
        self.graph = StateGraph(State)
        self.compiled = False
        self.provider = provider
        self.structured_output_method = struct_output_method
        self.debug_level = debug_level
        self.logging = logging
        self.model = model
        self.insert_sleep = insert_sleep
        self.last_run_time = -1
        self.usage_metadata = []
        self.prompt_config = {
            "generate_trace":None,
            "identify_sections":None,
            "understand_section":None,
            "check_balance":None,
            "classification":None,
            "verify_trace":None,
            "generate_trace_ideas":None
        } if not prompt_config else prompt_config
        self.load_prompts()
            
    def get_config(self):
        if self.provider == 'Ollama':
            return {
                "architecture": "COCSA",
                "set": self.cfg._set,
                "model": self.model,
                "temperature": self.cfg.temperature,
                "top_p":self.cfg.options["top_p"],
                "top_k":self.cfg.options["top_k"],
                "num_ctx": self.cfg.options["num_ctx"],
                "seed": self.cfg.seed,
                "prompt_v": self.cfg.prompt_config
            }
        else:
            #TODO: might add other architectures in the future
            return {
                "architecture": "COCSA",
                "model": self.model,
                "temperature": self.llm.temperature,
                "top_p": getattr(self.llm, "top_p", None),
                "top_k":getattr(self.llm, "top_k",None),
                "num_ctx": getattr(self.llm, "num_ctx", None),
                "seed" : getattr(self.llm, "seed", None)
            }
    

    def _get_concurrency_primitives(self, state : State, config : RunnableConfig, node_name : str = None):
        """The function describes the first node logic: 
        it runs a script to identify concurrency structures and functions using them"""        
        return {'concurrency_primitives': parse_go_concurrency(state['code'])}
    

    def _no_context_call(self, state : State, config: RunnableConfig, node_name : str, prompt:str, schema):
        """
        High order function to define a generic no_context node call.
        The function incapsulate the prompt inside a SystemMessage and then passes it to a structured llm 
        obtained with 'schema'.
        """
        if state['early_stop']:
            return "ABORTED"
        if self.insert_sleep:
            time.sleep(15)

        input = [SystemMessage(prompt)]
        if schema is not None:
            calling_llm = self.llm.with_structured_output(schema, method=self.structured_output_method, include_raw=True)
            if self.structured_output_method == 'json_mode':
                input += [HumanMessage(f"IMPORTANT: You must return a JSON object. "
            f"You MUST use exactly this template: {schema.model_json_schema()}. "
            "Do not include any other text or explanation.")]
        else:
            calling_llm = self.llm
        msg = try_to_invoke(calling_llm, input, schema, self.llm, execution_point=node_name)
        if self.debug_level > 2:
            print(msg)
        return msg


    @handle_early_exit('concurrency_sections')
    def _identify_sections(self, state : State, config: RunnableConfig, node_name : str = None):
        """
        The function calls the LLM for the 'identify_concurrency_sections' node.
        """
        sysprompt = self.identify_sections_prompt.format(code=state['code'], primitives=state['concurrency_primitives'])
        return self._no_context_call(state, config, node_name, sysprompt, schema=ConcurrencySections)
    
    @handle_early_exit("section_explanations")
    def _understand_section(self, state : WorkerState, config : RunnableConfig, node_name : str = None):
        """
        The function calls the llm for the 'understand_section' node.
        """
        sysprompt = self.understand_section_prompt.format(code=state['code'], section=state['section'])
        answer = self._no_context_call(state, config, node_name, sysprompt, schema=SectionExplanation)
        try:
            answer['parsed'] = [answer['parsed']]
        except:
            print(answer)
        return answer

    
    @handle_early_exit('balance_report')
    def _check_balancedness(self, state : State, config : RunnableConfig, node_name : str = None):
        """
        The function calls the llm for the 'check_balancedness' node.
        """
        sysprompt = self.check_balance_prompt.format(code=state['code'], primitives=state['concurrency_primitives'])
        return self._no_context_call(state, config, node_name, sysprompt, schema=BalanceReport)
    
    def _assign_workers(self, state : State, config: RunnableConfig, node_name : str = None):
        """
        The function assign to each LLM worker a concurrency section to analyze.
        """
        if state['early_stop'] or not state['concurrency_sections']:
            state['early_stop'] = True
            return []
        return [Send("understand_section", {"code":state['code'],"section":s, "early_stop":state['early_stop']}) for s in state['concurrency_sections']]


    @handle_early_exit("trace_ideas")
    def _orchestrate_traces(self, state: State, config:RunnableConfig, node_name:str = None):
        """
        The function calls the LLM for the 'trace_orchestrator' node.
        Effectively creating another orchestrator-workers workflow to handle the trace creations.
        """
        sysprompt = self.generate_trace_ideas_prompt.format(code=state['code'],sections=state['concurrency_sections'], understanding=state['section_explanations'], balanceanalysis=state['balance_report'])
        return self._no_context_call(state,config,node_name,sysprompt, schema=TraceIdeas)

    def _assign_trace_creators(self, state: State, config:RunnableConfig, node_name:str = None):
        """
        The function assigns to each LLM worker a concurrent possible trace idea to develop.
        """
        if state['early_stop'] or state['trace_ideas'] is None:
            return []
        if self.debug_level > 0:
            print(f"Trace ideas: {len(state['trace_ideas'].ideas)}")
        if len(state['trace_ideas'].ideas) > 5:
            state['trace_ideas'].ideas = state['trace_ideas'].ideas[:5]
            print(f"Trace ideas reduced to {state['trace_ideas']}")
        return [Send("generate_trace", {'code':state['code'],'trace_idea':idea, 'early_stop':state['early_stop']}) for idea in state['trace_ideas'].ideas]

    @handle_early_exit("trace_list")
    def _generate_trace(self, state : TraceCreatorState, config:RunnableConfig, node_name:str = None):
        """
        The function calls the LLM for the 'generate_trace' node.
        """
        sysprompt = self.generate_trace_from_idea_prompt.format(code=state['code'],trace_idea = state['trace_idea'])
        answer = self._no_context_call(state, config, node_name, sysprompt, schema=Trace)
        try:
            answer['parsed'] = [answer['parsed']]
        except:
            print(answer)
        return answer


    @handle_early_exit("trace_list")  
    def _generate_traces(self, state: State, config: RunnableConfig, node_name: str = None):
        """
        The function implements the logic of the 'generate_traces' node:
        it asks the llm to generate a list of problematic traces, 
        given the concurrency primitives identified.
        """
        sysprompt = self.generate_list_of_traces_prompt.format(primitives=state["concurrency_primitives"],code=state['code'])
        input = [SystemMessage(sysprompt)]
        structured_llm = self.llm.with_structured_output(Traces, method=self.structured_output_method, include_raw=True)
        msg = try_to_invoke(llm=structured_llm,msg=input,structured_output_schema=Traces,fixing_llm=self.llm,execution_point=node_name)
        return msg
    
    def _trace_selector(self, state:State):
        list = state["trace_list"]
        if list is None or len(list)==0:
            if self.debug_level > 0:
                print("Empty list")
                if list is None:
                    print("trace list is none")
            return {"active_trace":None}
        else:
            if self.debug_level > 0:
                print(f"Remaining traces: {len(list)}")
            active_trace = list[0]
            list = list[1:]
            return {
                "active_trace": active_trace, 
                "trace_list": Overwrite(list) # because the reducer 'add' is used for the list
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
        
        #input = [SystemMessage(self.verify_trace_prompt.format(code=state['code'],trace=state["active_trace"]))]
        #structured_llm = self.llm.with_structured_output(TraceEvaluation, method=self.structured_output_method, include_raw=True)
        #msg = try_to_invoke(llm=structured_llm,msg=input,structured_output_schema=TraceEvaluation,fixing_llm=self.llm,execution_point=node_name)
        sysprompt = self.verify_trace_prompt.format(code=state['code'],trace=state["active_trace"])
        return self._no_context_call(state, config, node_name, sysprompt, TraceEvaluation)


    @handle_early_exit("classification")
    def _create_classification(self, state : State, config: RunnableConfig, node_name: str = None):
        """The function describes the logic of the create classification node: 
        it asks the llm to classify the bug, given the program and the problematic trace."""

        #input = [SystemMessage(self.classification_prompt.format(code=state['code'],trace=state['active_trace'], trace_eval=state["trace_eval"]))]
        #structured_llm = self.llm.with_structured_output(BugClassification, method = self.structured_output_method, include_raw=True)
        #msg = try_to_invoke(llm=structured_llm,msg=input,structured_output_schema=BugClassification,fixing_llm=self.llm,execution_point=node_name)
        sysprompt = self.classification_prompt.format(code=state['code'],trace=state['active_trace'], trace_eval=state["trace_eval"])
        return self._no_context_call(state, config, node_name, sysprompt, BugClassification)
    
    def _empty_classification(self, state: State):
        """Function that describe the logic of the empty classification node: 
        since no problematic trace was found, there is no bug to report."""

        return {"classification" : {'type': None, 'subtype':None, 'cls':None}}
    
    def _early_termination(self, state : State):
        """Routing function to handle early stopping"""
        return "STOP" if state["early_stop"] else "NO STOP"

    def compile_chain(self, save_img=False):

        """The function compiles the chain, making the agent ready to be invoked."""
        if self.compiled:
            print("WARNING: you are recompiling an already compiled graph. It will be reinitialized")
            self.graph = StateGraph(State)
        
        # Adding the nodes in the graph
        self.graph.add_node("get_concurrency_primitives",self._get_concurrency_primitives)
        self.graph.add_node("identify_sections", self._identify_sections)
        self.graph.add_node("understand_section", self._understand_section)
        self.graph.add_node("check_balance", self._check_balancedness)
        self.graph.add_node("trace_orchestrator", self._orchestrate_traces)
        self.graph.add_node("generate_trace",self._generate_trace)
        self.graph.add_node("trace_selector", self._trace_selector)
        self.graph.add_node("check_trace",self._ask_if_trace_is_possible)
        self.graph.add_node("create_classification", self._create_classification)
        self.graph.add_node("empty_classification", self._empty_classification)

        # Connecting edges
        self.graph.add_edge(START, "get_concurrency_primitives")
        self.graph.add_edge("get_concurrency_primitives", "identify_sections")
        self.graph.add_conditional_edges(
            "identify_sections", self._assign_workers, ["understand_section"]
        )
        self.graph.add_edge("understand_section","check_balance")
        self.graph.add_edge("check_balance", "trace_orchestrator")
        self.graph.add_conditional_edges(
            "trace_orchestrator", self._assign_trace_creators, ["generate_trace"]
        )
        # TODO: maybe an intermediate node is needed for synchronization?
        # something between generate_trace and trace_selector
        self.graph.add_edge("generate_trace","trace_selector")
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

    def invoke_chain(self, code : str):
        """The function invokes the agent on the code provided."""
        if not self.compiled:
            print("WARNING: agent not compiled: you must call the 'compile_chain' method first.")
            return
        response = self.graph.invoke({"code":code, "early_stop": False, "tokens":(0,0)})
        if self.logging:
            log_tool_interactions(response)
        return response

    def load_prompts(self):
        self.generate_trace_from_idea_prompt = load_prompt_string(
            get_prompt_version(prompt_name="generate_trace", version=self.prompt_config["generate_trace"])
            )
        self.identify_sections_prompt = load_prompt_string(
            get_prompt_version("identify_sections", version=self.prompt_config["identify_sections"])
            )
        self.understand_section_prompt = load_prompt_string(
            get_prompt_version("understand_section", version=self.prompt_config["understand_section"])
        )
        self.check_balance_prompt = load_prompt_string(
            get_prompt_version("check_balance", version=self.prompt_config["check_balance"])
        )
        self.generate_trace_ideas_prompt = load_prompt_string(
            get_prompt_version("generate_trace_ideas", version=self.prompt_config["generate_trace_ideas"])
        )
        self.generate_list_of_traces_prompt =""
        self.verify_trace_prompt = load_prompt_string(
            get_prompt_version("verify_trace", version=self.prompt_config["verify_trace"])
        )
        self.classification_prompt = load_prompt_string(
            get_prompt_version("classification", version=self.prompt_config["classification"])
        )
        for key in self.prompt_config.keys():
            if self.prompt_config[key] is None:
                version = extract_latest_prompt_version(key)
                print(f"DEBUG: {version}")
                self.prompt_config[key] = extract_latest_prompt_version(key)

    def run_on_benchmark(self, paths_file : str, save_usage_metadata:bool=True):
        """The function runs the chain agent on all the programs specified by 'paths_file'.
        This must be a file containing all the paths to the go codes it is intended to analyze."""
        try:
            with open(paths_file, "r") as f:
                prg_paths = f.readlines()
        except Exception as e:
            print(f"Unable to read the paths_file: {e}")
            return 

        classification_data = {}
        thinking_log = {}
        verified_prg = 0
        start = time.time()
        input_tokens, output_tokens = 0,0
        for prg_path in prg_paths:
            # [:-1] to ignore the '\n'
            prg_path = prg_path[:-1]
            with open(prg_path, "r") as f:
                id = extract_id(prg_path)
                print(f"Id: {id}")
                prog = f.read()
                  
                response = self.invoke_chain(prog)
                try:
                    classification_data[id] = response['classification'].get_classification()
                except Exception as e:
                    print(f"Could not convert response into dictionary: {e}")
                    classification_data[id] = response['classification']
                    
                try:
                    intok,outtok = response['tokens']
                    input_tokens += intok
                    output_tokens += outtok
                except Exception as e:
                    print(f"Unable to extract token count: {e}")
                
                thinking_log[id] = response['reasoning']
                print(f"{response['classification']}")
                verified_prg+=1

                if save_usage_metadata:
                    self.usage_metadata += get_usage_metadata(response, verified_prg-1)

                print(f"Progress: {verified_prg}/{len(prg_paths)}")
                
                if self.insert_sleep:
                    print("-"*20+" Sleep inserted to avoid consuming all tokens "+"-"*20)
                    time.sleep(15)

        self.last_run_time = time.time()-start
        res = try_into_dataframe(classification_data, self.model)
        try:
            with open("results/thinking_"+self.model+".json", "w") as f:
                f.write(json.dumps(thinking_log, indent=4))
        except Exception as e:
            print(f"Cannot save thinking data: {e}")

        config = self.get_config()
        config["set"] = "validation" if "validation" in paths_file else "test"
        config["prompt_v"] = self.prompt_config
        return res, input_tokens, output_tokens, config
    

# llama-3.3-70b-versatile, qwen/qwen3-32b (No support tool calling), 
# moonshotai/kimi-k2-instruct, llama-3.1-8b-instant
# meta-llama/llama-4-scout-17b-16e-instruct, groq/compound-> no support for tool calling
# meta-llama/llama-4-maverick-17b-128e-instruct
# openai/gpt-oss-120b
if __name__ == '__main__':
    a = ChainOfDebugAgent(provider='Groq', model='llama-3.1-8b-instant',debug_level=2, insert_sleep=True)
    a.compile_chain(save_img=True)
    # Running the benchmark on the validation set
    print("Starting")
    df, input_tokens, output_tokens, config = a.run_on_benchmark("benchmarks_paths/validation_set.txt")
    print_token_count(a.usage_metadata)
    print(f"Time required:{a.last_run_time}")
    df = df.T  
    logger = ExperimentLogger()

    logger.log_run(config, df)
    #df.to_csv(f"results/benchmark_results_{a.model}.csv", index=True)

