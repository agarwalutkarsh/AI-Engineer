"""
AI Restaurant Agent (LangGraph + Groq SDK, no LangChain imports)

Flow:
    START -> show_menu -> ask_user -> agent
    agent  --tool call-->  take_order --ok-->  cook --ok--> serve --ok--> success -> END
                               |                 |              |
                               fail              fail           fail (retries left) -> cook
                               v                 v              fail (no retries)   -> apology -> END
                             agent (LLM explains & asks) -> ask_user -> agent ...
    agent  --no tool call (off-topic / question)--> ask_user
    Every customer turn counts as an attempt; after MAX_ORDER_ATTEMPTS -> goodbye -> END
"""

import copy
import json
import operator
import os
import random
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()  # searches upward from this file and finds Week4/.env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_ORDER_ATTEMPTS = 3      # customer turns allowed (off-topic turns count too)
MAX_RETRIES = 2             # re-cooks allowed after a failed serve
COOK_SUCCESS_RATE = 0.8
SERVE_SUCCESS_RATE = 0.85
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

MENU = {
    "Pizza":       {"price": 250, "qty": 5},
    "Pasta":       {"price": 200, "qty": 3},
    "Burger":      {"price": 150, "qty": 4},
    "Biryani":     {"price": 300, "qty": 2},
    "Cold Coffee": {"price": 120, "qty": 6},
}

client = Groq()  # reads GROQ_API_KEY from the environment


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    # Annotated + operator.add -> new messages are APPENDED (old + new)
    # Messages are plain dicts: {"role": "user" | "assistant" | "tool", "content": ...}
    messages: Annotated[list[dict], operator.add]
    # Plain types -> each node's returned value REPLACES the old one
    menu: dict[str, dict]       # {"Pizza": {"price": 250, "qty": 5}, ...}
    order: list[dict]           # [{"dish": "Pizza", "qty": 2, "price": 250}, ...]
    order_attempts: int
    retries: int
    status: str                 # new | chatting | order_ok | order_failed | cooked | cook_failed | served | serve_failed


# ---------------------------------------------------------------------------
# Tool schema (hand-written JSON schema in OpenAI/Groq format)
# Validation runs in the take_order node, which has access to the state.
# ---------------------------------------------------------------------------
TAKE_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "take_order",
        "description": (
            "Place the customer's food order. Call this whenever the customer names one "
            "or more dishes with quantities, e.g. '2 pizza and 1 pasta'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Dishes the customer wants",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dish": {"type": "string", "description": "Dish name exactly as written on the menu"},
                            "qty": {"type": "integer", "description": "Number of plates (positive integer)"},
                        },
                        "required": ["dish", "qty"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def format_menu(menu: dict) -> str:
    return "\n".join(
        f"  - {name:<12} Rs.{info['price']:>4}   ({info['qty']} available)"
        for name, info in menu.items()
    )


def system_prompt(menu: dict) -> str:
    return f"""You are a friendly restaurant ordering assistant.

MENU (dish, price, plates available):
{format_menu(menu)}

Rules:
- You ONLY help with ordering food from this menu: placing orders, menu items, prices, availability.
- For anything else (general knowledge, coding, jokes, etc.) politely refuse in one sentence and
  invite the customer to order from the menu.
- When the customer names dishes with quantities, call the `take_order` tool. Map their wording to the
  exact menu names (e.g. 'pizzas' -> 'Pizza', 'coffee' -> 'Cold Coffee'). If a quantity is missing, assume 1.
- If a dish is not on the menu, still pass it to the tool as written; the tool will reject it.
- Keep replies short, warm and customer-friendly. Never mention tools, nodes or internal systems."""


def call_llm(messages: list[dict], allow_tools: bool) -> dict:
    """Call Groq and convert the response into a plain dict message for the state."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[TAKE_ORDER_TOOL],
        tool_choice="auto" if allow_tools else "none",
        temperature=0,
    )
    msg = response.choices[0].message
    result = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in msg.tool_calls
        ]
    return result


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def show_menu(state: State) -> dict:
    text = (
        "Welcome to the AI Restaurant! Here's our menu:\n"
        f"{format_menu(state['menu'])}\n"
        "What would you like to order? (e.g. '2 pizza and 1 pasta')"
    )
    return {"messages": [assistant(text)]}


def ask_user(state: State) -> dict:
    # Graph pauses here. On resume, interrupt() returns the customer's reply.
    # Nothing with side effects goes ABOVE this line (the node re-runs on resume).
    reply = interrupt("Your reply")
    return {
        "messages": [{"role": "user", "content": str(reply)}],
        "order_attempts": state["order_attempts"] + 1,
    }


def agent(state: State) -> dict:
    status = state["status"]
    notes = []
    if status == "order_failed":
        notes.append({"role": "system", "content": (
            "The order could not be placed (see the latest tool result). Explain the problem kindly. "
            "If stock is short, offer the available quantity or suggest another dish, and ask the customer "
            "to confirm or change. Do NOT call any tool now."
        )})
    elif status == "cook_failed":
        dishes = ", ".join(f"{i['qty']} x {i['dish']}" for i in state["order"])
        notes.append({"role": "system", "content": (
            f"Kitchen update: cooking failed for the order ({dishes}). Apologise and ask whether the "
            "customer wants to repeat the same order or choose another dish. Do NOT call any tool now."
        )})

    messages = [{"role": "system", "content": system_prompt(state["menu"])}, *state["messages"], *notes]
    try:
        response = call_llm(messages, allow_tools=not notes)
    except Exception as exc:  # e.g. Groq tool-parse errors, network issues
        print(f"  (LLM error: {exc.__class__.__name__})")
        response = assistant("Sorry, I didn't quite catch that. Could you please repeat your order?")
    return {"messages": [response], "status": "chatting"}


def take_order_node(state: State) -> dict:
    last = state["messages"][-1]
    menu = state["menu"]
    lookup = {name.lower(): name for name in menu}

    requested: dict[str, int] = {}
    problems: list[str] = []

    for call in last["tool_calls"]:
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        for raw in args.get("items", []) or []:
            dish_text = str(raw.get("dish", "")).strip()
            name = lookup.get(dish_text.lower())
            try:
                qty = int(raw.get("qty", 1))
            except (TypeError, ValueError):
                qty = 0
            if name is None:
                problems.append(f"'{dish_text}' is not on the menu")
            elif qty <= 0:
                problems.append(f"invalid quantity for {name}")
            else:
                requested[name] = requested.get(name, 0) + qty

    for name, qty in requested.items():
        available = menu[name]["qty"]
        if available == 0:
            problems.append(f"{name} is sold out")
        elif qty > available:
            problems.append(f"customer asked for {qty} {name} but only {available} available")

    if not requested and not problems:
        problems.append("no dishes were specified")

    if problems:
        result, status, order = "ORDER REJECTED: " + "; ".join(problems), "order_failed", []
    else:
        order = [{"dish": n, "qty": q, "price": menu[n]["price"]} for n, q in requested.items()]
        result = "ORDER ACCEPTED: " + ", ".join(f"{i['qty']} x {i['dish']}" for i in order)
        status = "order_ok"

    # Every tool call must be answered with a "tool" message carrying its id
    new_msgs = [
        {"role": "tool", "tool_call_id": call["id"], "content": result}
        for call in last["tool_calls"]
    ]
    if status == "order_ok":
        new_msgs.append(assistant(f"[Order] Got it: {result.split(': ', 1)[1]}. Sending to the kitchen..."))
    return {"messages": new_msgs, "order": order, "status": status, "retries": 0}


def cook(state: State) -> dict:
    dishes = ", ".join(f"{i['qty']} x {i['dish']}" for i in state["order"])
    if random.random() < COOK_SUCCESS_RATE:
        return {"status": "cooked", "messages": [assistant(f"[Kitchen] {dishes} cooked successfully.")]}
    return {"status": "cook_failed", "messages": [assistant(f"[Kitchen] Something went wrong while cooking {dishes}.")]}


def serve(state: State) -> dict:
    if random.random() < SERVE_SUCCESS_RATE:
        menu = copy.deepcopy(state["menu"])
        for item in state["order"]:
            menu[item["dish"]]["qty"] -= item["qty"]   # stock decreases only on successful serve
        return {"status": "served", "menu": menu}

    retries = state["retries"] + 1
    msg = (
        f"[Waiter] Oops! The order was dropped while serving. Re-cooking (retry {retries}/{MAX_RETRIES})..."
        if retries <= MAX_RETRIES
        else "[Waiter] Oops! The order was dropped while serving again."
    )
    return {"status": "serve_failed", "retries": retries, "messages": [assistant(msg)]}


def success(state: State) -> dict:
    lines = [f"  {i['qty']} x {i['dish']:<12} Rs.{i['qty'] * i['price']}" for i in state["order"]]
    total = sum(i["qty"] * i["price"] for i in state["order"])
    text = (
        "Your order has been served. Enjoy your meal!\n"
        + "\n".join(lines)
        + f"\n  Total: Rs.{total}\nThank you for dining with us!"
    )
    return {"messages": [assistant(text)]}


def apology(state: State) -> dict:
    return {"messages": [assistant(
        "We're really sorry, we couldn't serve your order after several tries. "
        "You won't be charged. Please visit us again!"
    )]}


def goodbye(state: State) -> dict:
    return {"messages": [assistant(
        f"We've reached the limit of {MAX_ORDER_ATTEMPTS} attempts for this order session. "
        "Thank you for visiting, please come again!"
    )]}


# ---------------------------------------------------------------------------
# Conditional edges (decision functions)
# ---------------------------------------------------------------------------
def attempts_left(state: State) -> bool:
    return state["order_attempts"] < MAX_ORDER_ATTEMPTS


def route_after_agent(state: State) -> str:
    if state["messages"][-1].get("tool_calls"):
        return "take_order"
    return "ask_user" if attempts_left(state) else "goodbye"


def route_after_take_order(state: State) -> str:
    if state["status"] == "order_ok":
        return "cook"
    return "agent" if attempts_left(state) else "goodbye"


def route_after_cook(state: State) -> str:
    if state["status"] == "cooked":
        return "serve"
    return "agent" if attempts_left(state) else "goodbye"


def route_after_serve(state: State) -> str:
    if state["status"] == "served":
        return "success"
    return "cook" if state["retries"] <= MAX_RETRIES else "apology"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(State)
    builder.add_node("show_menu", show_menu)
    builder.add_node("ask_user", ask_user)
    builder.add_node("agent", agent)
    builder.add_node("take_order", take_order_node)
    builder.add_node("cook", cook)
    builder.add_node("serve", serve)
    builder.add_node("success", success)
    builder.add_node("apology", apology)
    builder.add_node("goodbye", goodbye)

    builder.add_edge(START, "show_menu")
    builder.add_edge("show_menu", "ask_user")
    builder.add_edge("ask_user", "agent")
    builder.add_conditional_edges("agent", route_after_agent,
                                  {"take_order": "take_order", "ask_user": "ask_user", "goodbye": "goodbye"})
    builder.add_conditional_edges("take_order", route_after_take_order,
                                  {"cook": "cook", "agent": "agent", "goodbye": "goodbye"})
    builder.add_conditional_edges("cook", route_after_cook,
                                  {"serve": "serve", "agent": "agent", "goodbye": "goodbye"})
    builder.add_conditional_edges("serve", route_after_serve,
                                  {"success": "success", "cook": "cook", "apology": "apology"})
    builder.add_edge("success", END)
    builder.add_edge("apology", END)
    builder.add_edge("goodbye", END)

    # MemorySaver stores the state after every node so interrupt() can pause/resume
    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Console runner
# ---------------------------------------------------------------------------
def print_new_messages(result: dict, already_printed: int) -> int:
    messages = result["messages"]
    for msg in messages[already_printed:]:
        if msg["role"] == "assistant" and msg.get("content"):
            print(f"\nAgent: {msg['content']}")
    return len(messages)


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state: State = {
        "messages": [],
        "menu": copy.deepcopy(MENU),
        "order": [],
        "order_attempts": 0,
        "retries": 0,
        "status": "new",
    }

    result = graph.invoke(initial_state, config)
    printed = 0
    while True:
        printed = print_new_messages(result, printed)
        if "__interrupt__" not in result:      # graph reached END
            break
        try:
            reply = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        result = graph.invoke(Command(resume=reply), config)

    print("\nRemaining stock:")
    print(format_menu(graph.get_state(config).values["menu"]))


if __name__ == "__main__":
    main()
