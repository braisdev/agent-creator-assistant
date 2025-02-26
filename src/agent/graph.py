from typing import Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, MessagesState

from agent.configuration import Configuration

from agent.nodes.message_manager import message_manager
from agent.nodes.update_expert import update_expert


# Load environment variables
load_dotenv()


def route_message(state: MessagesState) -> Literal[END, "update_expert"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "expert":
            return "update_expert"
        else:
            raise ValueError


# Define the StateGraph
workflow = StateGraph(MessagesState, config_schema=Configuration)

workflow.add_node("message_manager", message_manager)
workflow.add_node("update_expert", update_expert)
workflow.add_edge(START, "message_manager")
workflow.add_conditional_edges("message_manager", route_message)
workflow.add_edge("update_expert", "message_manager")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the workflow with the checkpointer
graph = workflow.compile(checkpointer=within_thread_memory, store=across_thread_memory)
