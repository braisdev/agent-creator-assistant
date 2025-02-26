import asyncio
from langmem import create_memory_manager
from pydantic import BaseModel, Field
from typing import Optional


class Expert(BaseModel):
    """
    Represents a custom Agent that we're calling 'Expert' various attributes.
    Attributes:
        name: Contains the attributes of Expert.
        description: Contains the description of the expert.
        instructions: Contains the instructions of the expert.
    """
    name: Optional[str] = Field(None, description="Name of Expert")
    description: Optional[int] = Field(None, description="Description of the Expert")
    instructions: Optional[str] = Field(None, description="Instructions for the System prompt of the Expert")


async def main():
    CUSTOM_EXPERT_INSTRUCTIONS = """You are a memory manager that focuses on capturing details about a custom 
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

    manager = create_memory_manager(
        "gpt-4o",
        instructions=CUSTOM_EXPERT_INSTRUCTIONS,
        schemas=[Expert]
    )
    conversation = [
        {"role": "assistant", "content": "What name do you want to give to your Expert?"},
        {"role": "user", "content": "My name is Antonio"},
        {"role": "assistant", "content": "Cool! nice to meet you Antonio, but what about the name of your Expert?"},
        {"role": "user", "content": "Ah! his name will be Anselmo"},
    ]
    memories = await manager(conversation)
    print(memories)  # First memory's content


asyncio.run(main())
