from typing import Literal
from pydantic import BaseModel


class ExpertFieldAssistantTool(BaseModel):
    """
    Tool used to generate or enhance content for a specific Expert field.
    """
    tool_type: Literal['help']
    field: Literal['name', 'description', 'instructions']
    content_hint: str = ""
