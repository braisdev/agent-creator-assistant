from typing import TypedDict, Literal


class ExpertFieldAssistantTool(TypedDict):
    """
    Tool used to generate or enhance content for a specific Expert field.

    Attributes:
    - tool_type: Must be "help" to signal this is a request for generation help.
    - field: Specifies which field we're generating help for ("name", "description", or "instructions").
    - content_hint: An optional string containing user-provided ideas, context, or partial content.
    """
    tool_type: Literal['help']
    field: Literal['name', 'description', 'instructions']
    content_hint: str
