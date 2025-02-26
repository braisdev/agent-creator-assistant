from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from agent.configuration import Configuration
from src.agent.tools.update_memory import UpdateMemory


def message_manager(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """
    Process the user's message, evaluate the current Expert schema stored in memory,
    and generate an appropriate, conversational response.

    This node performs the following tasks:

    1. Receives the user input from the state, which is then passed to the model.
    2. Retrieves the current Expert schema (which includes fields such as name, description,
       and instructions) from the provided store.
    3. Compares the incoming user message against the existing schema:
       - If the user provides an update (e.g., changing the agent's name from "Brais" to "Pepe"),
         the function will detect the difference and respond by acknowledging the change.
       - If a required field in the Expert schema is missing, the function will craft a casual,
         conversational prompt asking for that specific piece of missing information.
    4. Ensures that only one field is addressed per response, guiding the user step-by-step in
       completing the custom agent's profile.

    Args:
        state (MessagesState): The current message state containing the user's input.
        config (RunnableConfig): Configuration parameters that govern the node's execution.
        store (BaseStore): The persistent memory store holding the Expert schema details.

    Returns:
        dict: A dictionary containing a conversational response message.
    """

    model = ChatOpenAI(model="gpt-4o", temperature=0)

    SYSTEM_PROMPT = """You are a helpful chatbot.

    You are designed to assist the user in creating and updating a custom Expert profile.

    Your long-term memory holds the Expert profile, which consists of three key fields:
    1. **Name:** The Expert's name.
    2. **Description:** A brief overview of the Expert’s purpose and capabilities.
    3. **Instructions:** Detailed guidelines or system prompt instructions that define the Expert's behavior.

    Here is the current Expert Profile (may be empty if no information has been provided yet):
    <expert_profile>
    {expert_profile}
    </expert_profile>

    Instructions for processing user messages:

    1. **Evaluate the User's Input:**  
       Carefully read the user's message to determine if they are providing new or updated information related to the Expert profile.

    2. **Decide on Memory Updates:**  
       If the user's message contains any new or modified information about the Expert's Name, Description, or Instructions, update the corresponding field(s) by calling the UpdateMemory tool with type `expert`. If the message does not contain any Expert-related information, do not update memory.

    3. **Respond in a Conversational Style:**  
       - If an update is detected and performed, acknowledge the change by indicating the specific field that was updated (e.g., "I've updated the Expert's Name.").
       - If no update is necessary, simply respond naturally to the user's message.
       - Address one field at a time for clarity.

    4. **Ensure Profile Completeness:**  
       After processing the user's message, ensure the Expert profile is as complete as possible by guiding the user to provide any missing information.

    Respond naturally after any update, or if no update is necessary."""

    # Use the Configuration class to extract tenant_id and expert_id
    configuration = Configuration.from_runnable_config(config)
    tenant_id = configuration.tenant_id
    expert_id = configuration.expert_id
    # expert_profile = configuration.expert_profile

    # Retrieve profile memory from the store
    namespace = (tenant_id, expert_id)
    memories = store.search(namespace)
    if memories:
        expert_profile = memories[0].value
    else:
        expert_profile = None

    system_msg = SYSTEM_PROMPT.format(expert_profile=expert_profile)

    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": [response]}
