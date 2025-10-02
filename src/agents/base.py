"""
Base agent class for million agents social application
"""

from typing import Optional, Dict, Any
import uuid
import time


class BaseAgent:
    """Base agent class providing core functionality for all agents"""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize a base agent

        Args:
            agent_id: Unique identifier for the agent
            name: Display name for the agent
            role: Role or function of the agent
            description: Description of the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent_{self.agent_id[:8]}"
        self.role = role or "general"
        self.description = description or f"{self.name} is a {self.role} agent"
        self.created_at = time.time()
        self.last_active = time.time()

    def update_activity(self):
        """Update the last active timestamp"""
        self.last_active = time.time()

    def get_age(self) -> float:
        """Get the age of the agent in seconds"""
        return time.time() - self.created_at

    def get_idle_time(self) -> float:
        """Get how long the agent has been idle in seconds"""
        return time.time() - self.last_active

    def __str__(self) -> str:
        """String representation of the agent"""
        return (
            f"BaseAgent(id={self.agent_id}, "
            f"name={self.name}, "
            f"role={self.role})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the agent"""
        return (
            f"BaseAgent(agent_id='{self.agent_id}', "
            f"name='{self.name}', "
            f"role='{self.role}', "
            f"description='{self.description}', "
            f"created_at={self.created_at})"
        )

    def __eq__(self, other) -> bool:
        """Equality based on agent_id"""
        if not isinstance(other, BaseAgent):
            return False
        return self.agent_id == other.agent_id

    def __hash__(self) -> int:
        """Hash based on agent_id for use in sets/dicts"""
        return hash(self.agent_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "age": self.get_age(),
            "idle_time": self.get_idle_time()
        }

    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic agent information"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "age": self.get_age(),
            "idle_time": self.get_idle_time()
        }