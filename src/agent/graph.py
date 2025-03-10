from typing import Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

from agent.nodes.expert_field_assistant import expert_field_assistant
from agent.nodes.sync_profile import sync_profile
from agent.state import ExpertCreatorAssistant
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, MessagesState

from agent.configuration import Configuration

from agent.nodes.message_manager import message_manager
from agent.nodes.update_expert import update_expert

# Load environment variables
load_dotenv()


def route_message(state: ExpertCreatorAssistant) -> Literal[END, "update_expert", "expert_field_assistant"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END

    tool_call = message.tool_calls[0]
    args = tool_call.get('args', {})

    if args.get('update_type') == "expert":
        return "update_expert"
    elif args.get('tool_type') == "help":
        return "expert_field_assistant"
    else:
        raise ValueError(f"Unexpected tool call args: {args}")


# Define the StateGraph
workflow = StateGraph(ExpertCreatorAssistant, config_schema=Configuration)

workflow.add_node("sync_profile", sync_profile)
workflow.add_node("message_manager", message_manager)
workflow.add_node("update_expert", update_expert)
workflow.add_node("expert_field_assistant", expert_field_assistant)
workflow.add_edge(START, "sync_profile")
workflow.add_edge("sync_profile", "message_manager")
workflow.add_conditional_edges("message_manager", route_message)
workflow.add_edge("update_expert", "message_manager")
workflow.add_edge("expert_field_assistant", "message_manager")

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the workflow with the checkpointer
graph = workflow.compile(checkpointer=within_thread_memory)
