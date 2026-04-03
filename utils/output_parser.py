from langchain_classic.output_parsers import PydanticOutputParser, OutputFixingParser
from ollama import ResponseError as OllamaResponseError
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
import ast
from groq import BadRequestError
import re


def fix_common_hallucinations(text:str) -> str:
    """
    The function removes common LLM hallucinations that break JSON parsers.
    """
    if not isinstance(text, str):
        return text
    
    # 1. Strip Markdown code blocks (e.g., ```python or ```json)
    # This removes the starting tag and the ending backticks
    text = text.replace(r"```", "")
    
    # 2. Fix Pythonic Booleans and Nulls (True -> true, None -> null)
    # Using \b (word boundary) ensures we don't break words like "NoneEvent"
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    
    # 3. Fix Escaped Single Quotes (\' -> ')
    text = text.replace("\\'", "'")
    
    return text.strip()


def extract_raw_from_groq_exception(groq_err,execution_point: str = ""):
    
    raw_bad_text = None
    
    # Method A: Try to get the parsed error body directly from the Groq SDK
    if hasattr(groq_err, 'body') and isinstance(groq_err.body, dict):
        raw_bad_text = groq_err.body.get('error', {}).get('failed_generation')
    
    # Method B: Fallback string parsing (extracting the dict from the error message)
    if not raw_bad_text:
        # Splits "Error code: 400 - {'error': ...}" to grab just the dictionary string
        dict_str = str(groq_err).split(" - ", 1)[-1] 
        err_dict = ast.literal_eval(dict_str)
        raw_bad_text = err_dict['error']['failed_generation']

    print(f"[{execution_point}]: Salvaged text successfully. Passing to fixer...")
    return raw_bad_text

def extract_raw_from_generic_exception(parse_err, execution_point:str = ""):
    # Try to extract the failed generation
    # If it fails, the error is casted to a string so the fixer has context.
    raw_text = getattr(parse_err, 'llm_output', None)
    if not raw_text:
        raw_text = getattr(parse_err, 'content', str(parse_err))
    return raw_text


def try_removing_common_hallucinations(raw_bad_text, schema, execution_point: str = ""):
    """
    The function tries to apply the simple fix of removing escaped single quotes.\\
    The resulting text is passed through a PydanticOutputParser based on schema. In case of failure it returns None.
    """
    parser = PydanticOutputParser(pydantic_object=schema)
    try:
        clean = parser.parse(fix_common_hallucinations(raw_bad_text))
        return clean
    except Exception as e:
        print(f"[{execution_point}]: Removing common hallucinations wasn't enough. Fallback to OutputFixingParser.\n Exception: {e}")
        return None
    

def try_with_output_fixing_parser(raw_bad_text, schema, fixing_llm, execution_point:str = ""):
    """
    The function constructs an OutputFixingParser, based on the provided schema, and tries to fix the\\
    raw_bad_text in order to obtain a valid output. In case of failure, it returns None.
    """
    if not fixing_llm:
        return None
    
    parser = PydanticOutputParser(pydantic_object=schema)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm = fixing_llm)
    try:
        fixed_obj = fixing_parser.invoke(raw_bad_text)
        return fixed_obj
    except Exception as double_fault:
        print(f"[{execution_point}]: Fixing parser ALSO failed on error: {double_fault}")
        # Let it fall through to the generic abort logic
        return None


def try_to_invoke(llm, msg, structured_output_schema, fixing_llm = None, default_message: str = 'ABORTED', execution_point: str = ""):
    """
    The function tries to invoke the llm on the msg.
    In case the invocation fails it tries a two level fixing strategy:\\
        1. Some very common sources of errors are pythonic hallucinations and escaped single quotes.\\
           It removes both and tries to validate the resulting text; if it fails, it goes to method 2\\
        2. It uses an OutputFixingParser, meaning it calls the fixing_llm to rewrite the response in a valid output format.\\
    If both method fails it returns the default_message.
    """

    try:
        response = llm.invoke(msg)
        return response 
    except BadRequestError as groq_err:
        print(f"[{execution_point}]: Groq API rejected the tool call. Salvaging 'failed_generation'...")
        raw_bad_text = extract_raw_from_groq_exception(groq_err,execution_point)
        clean = try_removing_common_hallucinations(raw_bad_text, structured_output_schema, execution_point)
        if not clean and fixing_llm:
            clean = try_with_output_fixing_parser(raw_bad_text,structured_output_schema,fixing_llm,execution_point)
        if clean is not None:
            print(f"Returning fix from output parser.")
            return clean
        else:
            return default_message
        
    except (OutputParserException, ValidationError, OllamaResponseError) as parse_err:
        print(f"[{execution_point}]: Hard parse exception caught. Triggering fixing_parser... \nParse Error:{parse_err}")
        raw_bad_text = extract_raw_from_generic_exception(parse_err,execution_point)
        clean = try_removing_common_hallucinations(raw_bad_text, structured_output_schema, execution_point)
        if not clean and fixing_llm:
            clean = try_with_output_fixing_parser(raw_bad_text,structured_output_schema,fixing_llm,execution_point)
        if clean is not None:
            return clean
        else:
            return default_message
        
    #Fatal network/api exceptions
    except Exception as e:
        print(f"[{execution_point}]: Fatal error, case not covered.\n{e}")
        return default_message