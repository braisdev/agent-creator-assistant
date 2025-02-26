import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.graph import MessagesState
from langmem import create_memory_store_manager, create_memory_manager
from pydantic import BaseModel, Field
from typing import Optional

from agent.configuration import Configuration


class Expert(BaseModel):
    """
    Represents a custom Agent that we're calling 'Expert' with various attributes.
    Attributes:
        name: Contains the Expert's name.
        description: Contains the description of the Expert.
        instructions: Contains the instructions for the System prompt of the Expert.
    """
    name: Optional[str] = Field(None, description="Name of Expert")
    description: Optional[str] = Field(None, description="Description of the Expert")
    instructions: Optional[str] = Field(None, description="Instructions for the System prompt of the Expert")


_CUSTOM_EXPERT_INSTRUCTIONS = """You are a memory manager that focuses on capturing details about a custom 
    “Expert” the user is defining. The user may provide or modify the Expert’s name, description, and instructions (
    prompt). Your job is to store or update these details so that future interactions have an accurate, consistent 
    record of the Expert’s configuration.

    1. **Extract & Contextualize**  
       - Identify any references to the Expert’s name, description, and instructions within the conversation.
       - Note if the user is revising previous details or providing entirely new information.
       - Caveat uncertain or partial information with reasoning or confidence levels when needed.

    2. **Compare & Update**  
       - Check for changes or conflicts between the new details and any existing memory entries for this Expert.
       - Update or remove outdated information, ensuring that the final record remains consistent and correct.
       - Avoid storing redundant or contradictory statements—maintain a single, up-to-date source of truth for the Expert’s data.

    3. **Synthesize & Reason**  
       - Conclude what the final Expert name, description, and instructions should be based on the user’s latest inputs.
       - Combine or refine partial information if it clarifies the Expert’s role or capabilities.
       - Qualify conclusions if there is conflicting or incomplete data about the Expert.

    As the memory manager, record the Expert’s name, description, and instructions exactly as you want to recall them 
    in future conversations. Keep it concise, consistent, and free of contradictions, while retaining all important 
    details the user has provided."""


async def update_expert(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Use the Configuration class to extract tenant_id and expert_id
    configuration = Configuration.from_runnable_config(config)
    tenant_id = configuration.tenant_id
    expert_id = configuration.expert_id

    # Retrieve the current Expert profile from the store asynchronously
    namespace = (tenant_id, "experts", expert_id)
    memories = await store.asearch(namespace)

    conversation = state["messages"]

    manager = create_memory_store_manager(
        "gpt-4o",
        namespace=namespace,
        instructions=_CUSTOM_EXPERT_INSTRUCTIONS,
        schemas=[Expert],
        enable_inserts=False,
    )

    await manager.ainvoke({
        "messages": conversation,
        "existing": memories
    })

    # Retrieve tool call metadata from the last message
    tool_calls = state['messages'][-1].tool_calls

    return {
        "messages": [{
            "role": "tool",
            "content": "updated expert",
            "tool_call_id": tool_calls[0]['id']
        }]
    }
