from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from agent.state import ExpertCreatorAssistant

from langchain_core.messages import HumanMessage, AIMessage


def clean_chat_history(messages):
    return [
        msg for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage))
           and not msg.additional_kwargs.get('tool_calls')
    ]


def expert_field_assistant(state: ExpertCreatorAssistant):
    messages = state["messages"]

    help_field = ""
    tool_call_id = None
    for msg in reversed(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            help_field = msg.tool_calls[0]["args"]["field"]
            tool_call_id = msg.tool_calls[0]["id"]
            break

    # Specialized system prompt for field assistance.
    FIELD_HELP_PROMPT_TEMPLATE = """
    You are an expert content assistant for a custom "Expert" profile. The Expert profile consists of the following fields:
    
    - Name: A simple, clear identifier.
    - Description: A brief summary.
    - Instructions: A system prompt that guides the Expert’s behavior and must follow best practices.

    The user is asking for help with the "{help_field}" field.
    
    Your task:
    - For the "{help_field}" field, provide clear and actionable suggestions to generate or refine its content.
    {extra_instructions}
    
    Respond with a concise suggestion for the "{help_field}" field.
    """

    extra_instructions = ""

    if help_field == "instructions":
        extra_instructions = (
            "Ensure the proposed instructions adhere to best practices:\n"
            "1. Clarity and conciseness.\n"
            "2. A structured format with bullet points if needed.\n"
            "3. Inclusion of all necessary components to guide the Expert's behavior."
        )

    system_prompt = SystemMessage(
        "You are an AI subroutine that helps generate and refine the Expert's name, description, and instructions.")

    human_prompt = HumanMessage(FIELD_HELP_PROMPT_TEMPLATE.format(
        help_field=help_field,
        extra_instructions=extra_instructions
    ))

    prompt_template = ChatPromptTemplate.from_messages([
        system_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        human_prompt,
    ])

    llm = ChatOpenAI(model="gpt-4o")

    chain = (prompt_template | llm | StrOutputParser())

    # Invoke the chain with chat history
    response = chain.invoke({
        "chat_history": clean_chat_history(messages) # Pass the entire message history
    })

    return {
        "messages": [
            ToolMessage(
                content=response,
                tool_call_id=tool_call_id
            )
        ]
    }
