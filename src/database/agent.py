"""
Agent model for million-agent social platform
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from .config import Base


class Agent(Base):
    """Agent model representing an AI agent in the social network"""

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    personality_type = Column(String(50), nullable=False, default="balanced", index=True)

    # Big Five personality traits (0.0 to 1.0)
    openness = Column(Float, nullable=False, default=0.5)
    conscientiousness = Column(Float, nullable=False, default=0.5)
    extraversion = Column(Float, nullable=False, default=0.5)
    agreeableness = Column(Float, nullable=False, default=0.5)
    neuroticism = Column(Float, nullable=False, default=0.5)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    social_agent = relationship("SocialAgent", back_populates="agent", uselist=False)
    interactions_as_initiator = relationship("Interaction", foreign_keys="Interaction.initiator_id", back_populates="initiator")
    interactions_as_recipient = relationship("Interaction", foreign_keys="Interaction.recipient_id", back_populates="recipient")
    initiated_friendships = relationship("Friendship", foreign_keys="Friendship.initiator_id", back_populates="initiator")
    received_friendships = relationship("Friendship", foreign_keys="Friendship.recipient_id", back_populates="recipient")
    community_memberships = relationship("CommunityMembership", back_populates="agent")

    __table_args__ = (
        # Add check constraints for personality traits
        {"sqlite_autoincrement": True}
    )

    def __init__(self, name: str, personality_type: str = "balanced", **kwargs):
        """Initialize agent with validation"""
        self.name = name
        self.personality_type = personality_type

        # Validate personality type
        valid_types = ["balanced", "explorer", "builder", "connector", "leader", "innovator"]
        if personality_type not in valid_types:
            raise ValueError(f"Invalid personality_type: {personality_type}. Must be one of: {valid_types}")

        # Set personality traits with validation
        for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            value = kwargs.get(trait, 0.5)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{trait} must be between 0.0 and 1.0")
            setattr(self, trait, value)

    def __str__(self):
        """String representation of agent"""
        return f"Agent(id={self.id}, name='{self.name}', personality='{self.personality_type}')"

    def __repr__(self):
        """Detailed representation of agent"""
        return (f"Agent(id={self.id}, name='{self.name}', personality='{self.personality_type}', "
                f"openness={self.openness}, conscientiousness={self.conscientiousness}, "
                f"extraversion={self.extraversion}, agreeableness={self.agreeableness}, "
                f"neuroticism={self.neuroticism})")

    def update_personality_traits(self, **kwargs):
        """Update personality traits with validation"""
        for trait, value in kwargs.items():
            if trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{trait} must be between 0.0 and 1.0")
                setattr(self, trait, value)
                self.updated_at = datetime.utcnow()

    def get_personality_summary(self) -> dict:
        """Get personality traits summary"""
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism
        }