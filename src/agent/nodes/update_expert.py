import logging

from langmem import create_memory_manager
from pydantic import BaseModel, Field
from typing import Optional

from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

from agent.state import ExpertCreatorAssistant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Expert(BaseModel):
    """
    Represents a custom Agent called 'Expert' with various attributes.
    Attributes:
        name: The Expert's name.
        description: A brief description of the Expert.
        instructions: Detailed instructions for the Expert.
    """
    name: Optional[str] = Field(None, description="Name of Expert")
    description: Optional[str] = Field(None, description="Description of the Expert")
    instructions: Optional[str] = Field(None, description="Instructions for the Expert")


# Updated instructions with placeholders for recent messages and the current expert profile from state.
_CUSTOM_EXPERT_INSTRUCTIONS = """You are a memory manager that focuses on capturing details about a custom "Expert" the user is defining.

Recent Messages:
{recent_messages}

Current Expert Profile:
{current_expert_profile}

Your task is to update the Expert profile based solely on these inputs. Do not consider any older conversation context.

Steps:
1. Review the Recent Messages to identify any changes or updates to the Expert’s name, description, or instructions.
2. Merge these changes with the Current Expert Profile.
3. Output a single, updated Expert profile record that is concise, consistent, and free of contradictions.
"""


async def update_expert(state: ExpertCreatorAssistant):

    # Get the current expert profile from state (synchronized earlier via sync_profile).
    current_expert_profile = state.get("expert_profile", {})

    # Find the last HumanMessage, last AIMessage, and last ToolMessage in the conversation.
    last_human_message = None
    last_ai_messages = []

    for msg in reversed(state["messages"]):
        if not last_human_message and isinstance(msg, HumanMessage):
            last_human_message = msg
        if isinstance(msg, AIMessage):
            if len(last_ai_messages) < 3:
                last_ai_messages.append(msg)
        if last_human_message and len(last_ai_messages) == 3:
            break

    # Reverse the AI messages so they appear in chronological order.
    last_ai_messages = list(reversed(last_ai_messages))

    # Combine the contents of the three messages (if they exist) into a single string.
    recent_messages_parts = []
    if last_human_message:
        recent_messages_parts.append("Human: " + last_human_message.content)
    for ai_msg in last_ai_messages:
        recent_messages_parts.append("AI: " + ai_msg.content)

    recent_messages_str = "\n\n".join(recent_messages_parts)

    # Convert the current expert profile from state to a string representation.
    current_expert_profile_str = str(current_expert_profile)

    # Format the instructions with the recent messages and current expert profile.
    optimized_instructions = _CUSTOM_EXPERT_INSTRUCTIONS.format(
        recent_messages=recent_messages_str,
        current_expert_profile=current_expert_profile_str
    )

    # Create a memory manager with the optimized instructions.
    manager = create_memory_manager(
        "gpt-4o",
        instructions=optimized_instructions,
        schemas=[Expert],
    )

    # Prepare input: use the last HumanMessage and the last two AIMessage objects (if available).
    input_messages = []
    if last_human_message is not None:
        input_messages.append(last_human_message)
    input_messages.extend(last_ai_messages)
    logger.info(f"Input messages: {input_messages}")

    input_data = {
        "messages": input_messages,
    }

    # Invoke the memory manager.
    profile_expert = await manager.ainvoke(input_data)

    if profile_expert and len(profile_expert) > 0:
        updated_expert = profile_expert[0][1]  # Extract the Expert instance.
        expert_profile_value = updated_expert.dict()  # Convert to dict.
    else:
        expert_profile_value = Expert()

    # Retrieve tool call metadata from the most recent message that has tool_calls.
    tool_call_id = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_id = msg.tool_calls[0]["id"]
            break

    # Return a Command object that updates the state.
    return Command(
        update={
            "expert_profile": expert_profile_value,
            "messages": [
                ToolMessage(
                    content="updated expert",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
