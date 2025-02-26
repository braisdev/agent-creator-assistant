"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, fields, field
from typing import Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # by default, we set the language as the language platform
    tenant_id: str = "brais"
    expert_id: str = "experto_de_brais"
    expert_profile: dict = field(default_factory=lambda: {
        "name": None,
        "description": None,
        "instructions": None,
    })

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
