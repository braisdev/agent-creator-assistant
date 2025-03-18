from typing import Literal
from pydantic import BaseModel


# Update memory tool as a BaseModel instead of TypedDict
class UpdateMemory(BaseModel):
    """ Decision on what memory type to update """
    update_type: Literal['expert']
