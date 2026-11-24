from typing import TypedDict
from langgraph.graph import END, StateGraph

# State
class State(TypedDict):
    number: int

# Node 1
def doubleIt(state: State) -> dict:
    stateNumber = state["number"]
    newNumber = stateNumber * 2
    print(f"Double It Function")
    print(newNumber)
    return {
        "number" : newNumber
    }


# Node 2
def finishIt(state: State) -> dict:
    stateNumber = state["number"]
    print("Finish It funtion")
    print(stateNumber)
    return {
        "number" : stateNumber
    }


# Normal Funtion
def decisionMake(state : State) -> str:
    if state["number"] < 100:
        return "double"
    return "finish"


# Graph Initiallize

builder = StateGraph(State)

# Adding Node
builder.add_node("doubleIt", doubleIt)
builder.add_node("finishIt", finishIt)

# Set Entry Point
builder.set_entry_point("doubleIt")

# Add conditional Edge - source, which function to make decision, which node on what decision
builder.add_conditional_edges("doubleIt", decisionMake, {"double": "doubleIt", "finish": "finishIt"})

# it adds an edge from double to decision and checks if the number is greater than 100, if no then it would call the double function again. if yes, then it would call the finish node

builder.add_edge("finishIt", END)

# Create the graph
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"number" : 5})
    print(result)