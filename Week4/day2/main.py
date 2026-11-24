"""
MATH & WEB SEARCH AI AGENT  -  simple explanation (read this first!)
=====================================================================

WHAT THIS PROGRAM DOES
----------------------
You type a question. An LLM (Groq) answers it. If the LLM needs help,
it can use 2 tools that WE wrote in Python:
    1. calculate(expression) -> does math,       e.g. "0.15 * 2400" -> 360
    2. web_search(query)     -> searches the web (Tavily), e.g. "PM of Japan"

THE MOST IMPORTANT IDEA
-----------------------
The LLM can NOT run Python code. It only READS the tool descriptions
(the `tools` JSON list) and then ASKS our code to run a tool.
Our code runs the tool and gives the result back to the LLM.

WHAT IS A TOOL CALL?
--------------------
A tool call is a "request note" from the LLM saying:
"please run THIS tool with THESE inputs for me".
Think of it like an ORDER SLIP in a restaurant:

    tool_call.id                 = "call_abc123"   -> order number
    tool_call.function.name      = "calculate"     -> which dish (tool)
    tool_call.function.arguments = '{"expression": "0.15 * 2400"}'
                                                    -> the inputs (a JSON STRING,
                                                       convert with json.loads)

WHAT IS tool_call.id?
---------------------
- It is NOT the id of your question.
- It is NOT the id of a loop round.
- It IS the id of ONE tool request. Every tool request gets a new id.
When we send a result back, we attach the SAME id ("tool_call_id"),
so the LLM knows which request the result belongs to.
This matters when the LLM asks for 2+ tools at the same time, e.g.:
    call_A -> web_search("population of Japan")  -> result for call_A
    call_B -> web_search("population of India")  -> result for call_B

QUESTION vs ROUND vs TOOL CALL
------------------------------
    Question  = what you typed. 1 per run_agent() call. No id.
    Round     = 1 trip to the LLM. Can be several (max MAX_TOOL_ROUNDS). No id.
    Tool call = 1 request to run a tool. Can be several per round. HAS an id.

THE FLOW, STEP BY STEP (example)
--------------------------------
Question: "What is the population of Japan divided by 1000?"

  Round 1:  We send: question + tool descriptions (tool_choice="auto")
            LLM replies with NO answer, only a tool call:
                id="call_1", name="web_search", args={"query": "population of Japan"}
            Our code runs web_search() -> "about 123,000,000"
            We send back: {"role": "tool", "tool_call_id": "call_1", "content": ...}

  Round 2:  LLM reads the result, still needs to divide -> another tool call:
                id="call_2", name="calculate", args={"expression": "123000000 / 1000"}
            Our code runs calculate() -> 123000.0
            We send back: {"role": "tool", "tool_call_id": "call_2", "content": ...}

  Round 3:  LLM has everything. It returns NO tool calls, only text:
                "Japan's population is about 123 million; divided by 1000 = 123,000."
            -> This is the final answer, so run_agent() returns it.

HOW THE CODE KNOWS IT'S THE FINAL ANSWER
----------------------------------------
    message.tool_calls is empty -> LLM answered directly -> return message.content
    message.tool_calls has items -> run those tools, send results, loop again

tool_choice OPTIONS
-------------------
    "auto"     -> LLM decides by itself whether to use a tool (used here)
    "none"     -> LLM is not allowed to use tools
    "required" -> LLM must use a tool

"""

import os
import json
import ast
import math
import operator

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL")
TAVILY_API_KEY = os.getenv("TAVILY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
if not MODEL:
    raise ValueError("MODEL not found in .env")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY not found in .env")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

MAX_TOOL_ROUNDS = 5


# ---------------------------------------------------------------------------
# Tool 1: calculate (safe evaluation, no eval())
# ---------------------------------------------------------------------------
BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return BINARY_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_evaluate_node(node.operand))

    if isinstance(node, ast.Name) and node.id in ALLOWED_CONSTANTS:
        return ALLOWED_CONSTANTS[node.id]

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ALLOWED_FUNCTIONS
        and not node.keywords
    ):
        args = [_evaluate_node(arg) for arg in node.args]
        return ALLOWED_FUNCTIONS[node.func.id](*args)

    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> dict:
    """Safely evaluate a math expression and return the result."""
    try:
        expression = expression.replace("^", "**")
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_node(tree)
        return {"expression": expression, "result": result}
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero"}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: web_search (Tavily)
# ---------------------------------------------------------------------------
def web_search(query: str) -> dict:
    """Search the web using Tavily and return a short answer with sources."""
    try:
        response = tavily_client.search(
            query=query,
            max_results=3,
            include_answer=True,
        )
        sources = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": (item.get("content") or "")[:500],
            }
            for item in response.get("results", [])
        ]
        return {"query": query, "answer": response.get("answer"), "sources": sources}
    except Exception as e:
        return {"query": query, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool definitions (JSON schema sent to the LLM)
# ---------------------------------------------------------------------------
# HOW TOOL CALLING WORKS
# The LLM can NOT run Python code itself. Instead:
#   1. We send the LLM a description of each tool in JSON (name, what it does,
#      what arguments it takes). This is the "tools" list below.
#   2. Based on the user's question, the LLM decides whether a tool is needed.
#      If yes, instead of answering, it replies with a "tool call", e.g.
#         {"name": "calculate", "arguments": "{\"expression\": \"2+2\"}"}
#   3. OUR code reads that tool call, runs the real Python function,
#      and sends the result back to the LLM as a message with role "tool".
#   4. The LLM reads the result and either calls another tool or writes
#      the final answer for the user.
#
# - "description" is what the LLM reads to decide WHEN to use the tool,
#   so it should be clear and specific.
# - "parameters" is a JSON Schema that tells the LLM WHAT arguments to send
#   and their types. "required" lists arguments that must always be present.
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression and return the numeric result. "
                "Use this for ANY arithmetic or math calculation. Supports + - * / // % **, "
                "parentheses, and functions sqrt, sin, cos, tan, log, log10, exp, abs, round, "
                "and constants pi and e."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '(25 * 4) + sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current events, facts, or any information "
                "you do not know. Returns a short answer and source links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'current population of India'",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# Maps the tool name the LLM sends back -> the actual Python function to run.
available_functions = {
    "calculate": calculate,
    "web_search": web_search,
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to two tools. "
    "Use the 'calculate' tool for any math calculation instead of computing it yourself. "
    "Use the 'web_search' tool for facts, current events, or anything you are unsure about. "
    "You may call tools multiple times, e.g. search first and then calculate with the result. "
    "Give a clear, concise final answer."
)


def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    # Loop because the LLM may need several tool calls in a row
    # (e.g. web_search to find a number, then calculate with it).
    # MAX_TOOL_ROUNDS stops it from looping forever.
    for _ in range(MAX_TOOL_ROUNDS):
        # tools=tools        -> tells the LLM which tools exist
        # tool_choice="auto" -> the LLM decides by itself whether to call a
        #                       tool or answer directly ("none" would disable
        #                       tools, "required" would force a tool call)
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message

        # message.tool_calls is a list of tools the LLM wants us to run.
        # If it is empty, the LLM has answered directly -> that's the final answer.
        if not message.tool_calls:
            return message.content

        # WHAT IS A TOOL CALL? (simple explanation)
        # The LLM cannot run calculate() or web_search() itself. When it needs
        # a tool, it does NOT answer the question. It sends back a "request
        # note" saying: "please run THIS tool with THESE inputs for me".
        # That request note is a tool call.
        #
        # message.tool_calls is a LIST of these notes (the LLM can ask for
        # more than one tool at once). Each tool_call looks like this:
        #
        #   tool_call.id                 = "call_abc123"
        #   tool_call.type               = "function"
        #   tool_call.function.name      = "calculate"
        #   tool_call.function.arguments = '{"expression": "0.15 * 2400"}'
        #
        #   - id:        a unique ticket number made by the LLM for this request.
        #                When we send the result back, we attach the same id so
        #                the LLM knows which request the result belongs to
        #                (important when it asked for 2+ tools at once).
        #   - name:      WHICH tool to run. It matches the "name" we gave in the
        #                tools list, so we can find it in available_functions.
        #   - arguments: the INPUTS for the tool. It is a JSON *string*, not a
        #                dict, so we must convert it with json.loads().
        #
        # Analogy: a restaurant order slip.
        #   id = order number, name = the dish, arguments = "extra cheese".
        #   The kitchen (our code) cooks it and returns it with the SAME
        #   order number, so the waiter (LLM) knows whose order it is.
        #
        # First, add the LLM's tool-call request to the history. The API
        # requires this assistant message to come before the tool results.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        # Run every tool the LLM asked for (it can ask for several at once).
        for tool_call in message.tool_calls:
            # Look up the real Python function by the name the LLM gave
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)

            # Arguments arrive as a JSON string -> convert to a Python dict
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            print(f"  -> Tool call: {function_name}({arguments})")

            if function_to_call is None:
                result = {"error": f"Unknown tool: {function_name}"}
            else:
                try:
                    # **arguments unpacks the dict, so {"expression": "2+2"}
                    # becomes calculate(expression="2+2")
                    result = function_to_call(**arguments)
                except TypeError as e:
                    result = {"error": f"Invalid arguments: {e}"}

            # Send the result back to the LLM with role "tool".
            # tool_call_id links this result to the request it answers.
            # content must be a string, so the dict is converted to JSON.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )
        # Loop again: the LLM now sees the tool results and continues.

    return "Stopped: too many tool calls without a final answer."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Math & Web Search Agent (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        print(f"Agent: {run_agent(user_input)}")
