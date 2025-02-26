from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field


@dataclass
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