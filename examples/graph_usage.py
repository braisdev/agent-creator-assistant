from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.agent.graph import graph
import logging

logging.disable(logging.CRITICAL)

load_dotenv()


def main():

    # Initial configuration for the agent
    config = {
        "configurable": {
            "thread_id": "1",
            "tenant_id": "brais",
            "expert_id": "experto_de_brais",
            "expert_profile": {
                "name": None,
                "description": None,
                "instructions": None,
            }
        }
    }

    # Initialize the conversation state
    conversation_state = {"messages": []}

    print("Interactive Chatbot Session. Type 'exit' to quit.\n")

    while True:
        # Ask if you want to update the expert_profile configuration
        update_profile = input("Do you want to modify the expert_profile? (yes/no): ")
        if update_profile.strip().lower() in ["yes", "y"]:
            print("Update expert_profile:")
            new_name = input(f"Enter expert name (current: {config['configurable']['expert_profile']['name']}): ")
            if new_name.strip():
                config["configurable"]["expert_profile"]["name"] = new_name.strip()

            new_description = input(
                f"Enter expert description (current: {config['configurable']['expert_profile']['description']}): ")
            if new_description.strip():
                config["configurable"]["expert_profile"]["description"] = new_description.strip()

            new_instructions = input(
                f"Enter expert instructions (current: {config['configurable']['expert_profile']['instructions']}): ")
            if new_instructions.strip():
                config["configurable"]["expert_profile"]["instructions"] = new_instructions.strip()

        # Get your chat message
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break

        # Append the human message to the conversation
        conversation_state["messages"].append(HumanMessage(content=user_input))

        # Stream the agent's response, passing the config
        print("Agent: ", end="", flush=True)
        for msg, metadata in graph.stream(conversation_state, stream_mode="messages", config=config):
            if metadata["langgraph_node"] == "message_manager":
                print(msg.content, end="", flush=True)
        print()  # New line after the message is complete


if __name__ == "__main__":
    main()