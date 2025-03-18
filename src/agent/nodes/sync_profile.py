from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
import logging

from agent.configuration import Configuration
from agent.state import ExpertCreatorAssistant, Expert

# Configure logging
logging.basicConfig(level=logging.INFO)


def sync_profile(state: ExpertCreatorAssistant, config: RunnableConfig):
    """
    Comprehensively synchronize the expert profile from the configuration.

    This node ensures that the graph's state is aligned with the latest
    configuration, which could have been updated by the front-end.
    Any missing field is set to "NOT SET".
    """
    # Extract configuration
    configuration = Configuration.from_runnable_config(config)
    config_profile = configuration.expert_profile

    # Ensure missing fields are explicitly set to "NOT SET"
    synced_profile = {
        "name": config_profile.get("name") or "NOT SET",
        "description": config_profile.get("description") or "NOT SET",
        "instructions": config_profile.get("instructions") or "NOT SET"
    }
    synced_profile = Expert(**synced_profile).model_dump()

    # Get the current state's expert profile (if exists)
    current_profile = state.get('expert_profile', Expert().model_dump())
    logging.info(f"Current profile: {current_profile}")

    # Track changes between the current profile and the new synced profile
    changes = []
    if current_profile.get("name") != synced_profile.get("name"):
        changes.append(
            f"Expert's name changed from '{current_profile.get('name', 'NOT SET')}' to '{synced_profile.get('name', 'NOT SET')}'"
        )
    if current_profile.get("description") != synced_profile.get("description"):
        changes.append(
            f"Expert's description updated from '{current_profile.get('description', 'NOT SET')}' to '{synced_profile.get('description', 'NOT SET')}'"
        )
    if current_profile.get("instructions") != synced_profile.get("instructions"):
        changes.append("Expert's instructions have been modified")

    # Construct a string representing the current state of the expert
    current_state = (
        f"Name: {synced_profile.get('name', 'NOT SET')}\n"
        f"Description: {synced_profile.get('description', 'NOT SET')}\n"
        f"Instructions: {synced_profile.get('instructions', 'NOT SET')}"
    )

    # Prepare sync message including changes and current expert state
    if changes:
        sync_message = AIMessage(
            content="Profile synchronization detected:\n" +
                    "\n".join(f"• {change}" for change in changes) +
                    "\n\nCurrent Expert State:\n" + current_state
        )
    else:
        sync_message = AIMessage(
            content="Profile synchronized. No changes detected.\n\nCurrent Expert State:\n" + current_state
        )

    # Return the synchronized profile and a sync message
    return {
        "expert_profile": synced_profile,
        "messages": [sync_message]
    }
