import logging
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import ExpertCreatorAssistant, Expert
from agent.tools.expert_field_assistant_tool import ExpertFieldAssistantTool
from src.agent.tools.update_memory import UpdateMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def message_manager(state: ExpertCreatorAssistant, config: RunnableConfig):
    """
    Process the user's message using the synchronized expert profile stored in state.
    It uses the profile as provided by sync_profile, which guarantees that missing fields
    are set to "NOT SET". The system prompt then instructs the LLM to use ONLY those values.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    SYSTEM_PROMPT = """You are a helpful chatbot.

You are designed to assist the user in creating and updating a custom Expert profile.

⚠️ CRITICAL SOURCE OF TRUTH ⚠️
Current Expert Profile:
Name: {name}
Description: {description}
Instructions: {instructions}

⚠️ IMPORTANT RULE: The above values are the ONLY accurate and current values for the Expert Profile.
COMPLETELY IGNORE any references to profile values in previous conversation history.
If any field in the Current Expert Profile is "NOT SET", consider that field as unset and NEVER fill it in using historical data.

If asked about the Expert's profile:
- If Name is "NOT SET", you MUST state that the Expert does not have a name yet.
- If Description is "NOT SET", you MUST state that the Expert does not have a description yet.
- If Instructions are "NOT SET", you MUST state that the Expert does not have instructions yet.

DO NOT reference any old values mentioned in previous messages.

Instructions for processing user messages:

1. Evaluate the user's input to determine if new or updated information is provided regarding the Expert profile.
2. If updates are provided, update the corresponding field(s) by calling the UpdateMemory tool with type `expert`.
3. Respond naturally to the user's message, addressing only one field at a time.
4. When referring to the Expert profile, ONLY use the values provided in the "Current Expert Profile" section above.
5. If the user asks for help generating the content for any of the Expert's attributes (name, description, or instructions), call the **ExpertFieldAssistant** tool with type `help`.
"""

    # Get the expert profile from state (already synchronized by sync_profile)
    expert_data = state.get("expert_profile", {})
    expert_profile = Expert(**expert_data)

    system_msg = SYSTEM_PROMPT.format(
        name=expert_profile.name,
        description=expert_profile.description,
        instructions=expert_profile.instructions
    )

    logger.info(
        f"Current Expert Profile - Name: {expert_profile.name}, Description: {expert_profile.description}, Instructions: {expert_profile.instructions}")
    logger.info(f"Processing user message: {state['messages'][-1].content if state['messages'] else 'No messages'}")

    # Invoke the model with the system prompt as the sole source of truth plus conversation history.
    response = model.bind_tools([UpdateMemory, ExpertFieldAssistantTool], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    logger.info(f"Model response: {response.content}")

    return {
        "messages": [response],
        "expert_profile": state["expert_profile"]
    }
