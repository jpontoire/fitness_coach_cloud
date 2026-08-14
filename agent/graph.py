from langgraph.graph import StateGraph, END
from .state import AgentState
from .router import router_node
from .nodes import load_resources, rag_node, generate_node, refusal_node, program_node

def build_graph():
    resources = load_resources()

    def rag_node_wrapped(state):
        return rag_node(state, resources)

    def generate_node_wrapped(state):
        return generate_node(state, resources)

    def refusal_node_wrapped(state):
        return refusal_node(state, resources)

    def program_node_wrapped(state):
        return program_node(state, resources)

    def route_decision(state):
        intent = state.get("intent", "conversational")
        if intent == "exercise_lookup":
            return "rag"
        elif intent == "off_topic":
            return "refusal"
        elif intent == "program_request":
            return "program"
        else:
            return "generate"

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node_wrapped)
    graph.add_node("generate", generate_node_wrapped)
    graph.add_node("refusal", refusal_node_wrapped)
    graph.add_node("program", program_node_wrapped)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "generate": "generate",
            "refusal": "refusal",
            "program": "program",
        }
    )

    graph.add_edge("rag", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("refusal", END)
    graph.add_edge("program", END)

    return graph.compile()
