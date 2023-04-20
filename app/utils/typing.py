from typing import TypedDict, List


class AgentConfig(TypedDict):
    display_name: str
    type: str
    path: str


class EnvConfig(TypedDict):
    path: str
    env_id: str
    env_name: str
    description: str
    agents: List[AgentConfig]
