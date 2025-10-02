"""
SocialAgent model for million-agent social platform
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from .config import Base


class SocialAgent(Base):
    """SocialAgent model extending Agent with social features"""

    __tablename__ = "social_agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, unique=True, index=True)

    # Social profile
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)

    # Reputation and activity metrics
    reputation_score = Column(Float, nullable=False, default=50.0)  # 0-100
    activity_level = Column(Float, nullable=False, default=0.5)  # 0-1

    # Social preferences
    social_preference = Column(String(20), nullable=False, default="balanced", index=True)
    communication_style = Column(String(20), nullable=False, default="neutral", index=True)

    # Counters for social statistics
    friends_count = Column(Integer, nullable=False, default=0)
    interactions_count = Column(Integer, nullable=False, default=0)
    communities_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    agent = relationship("Agent", back_populates="social_agent")

    __table_args__ = (
        UniqueConstraint('agent_id', name='uq_social_agent_agent_id'),
        {"sqlite_autoincrement": True}
    )

    def __init__(self, agent_id: int, **kwargs):
        """Initialize social agent with validation"""
        self.agent_id = agent_id

        # Validate and set social preference
        valid_preferences = ["balanced", "extroverted", "introverted", "selective", "explorer"]
        social_preference = kwargs.get('social_preference', 'balanced')
        if social_preference not in valid_preferences:
            raise ValueError(f"Invalid social_preference: {social_preference}. Must be one of: {valid_preferences}")
        self.social_preference = social_preference

        # Validate and set communication style
        valid_styles = ["neutral", "friendly", "formal", "casual", "professional", "humorous"]
        communication_style = kwargs.get('communication_style', 'neutral')
        if communication_style not in valid_styles:
            raise ValueError(f"Invalid communication_style: {communication_style}. Must be one of: {valid_styles}")
        self.communication_style = communication_style

        # Validate and set reputation score
        reputation_score = kwargs.get('reputation_score', 50.0)
        if not 0.0 <= reputation_score <= 100.0:
            raise ValueError(f"reputation_score must be between 0.0 and 100.0")
        self.reputation_score = reputation_score

        # Validate and set activity level
        activity_level = kwargs.get('activity_level', 0.5)
        if not 0.0 <= activity_level <= 1.0:
            raise ValueError(f"activity_level must be between 0.0 and 1.0")
        self.activity_level = activity_level

        # Set other attributes
        self.bio = kwargs.get('bio')
        self.avatar_url = kwargs.get('avatar_url')
        self.friends_count = kwargs.get('friends_count', 0)
        self.interactions_count = kwargs.get('interactions_count', 0)
        self.communities_count = kwargs.get('communities_count', 0)

    def __str__(self):
        """String representation of social agent"""
        agent_name = self.agent.name if self.agent else f"Agent({self.agent_id})"
        return f"SocialAgent(agent='{agent_name}', preference='{self.social_preference}', reputation={self.reputation_score})"

    def __repr__(self):
        """Detailed representation of social agent"""
        agent_name = self.agent.name if self.agent else f"Agent({self.agent_id})"
        return (f"SocialAgent(id={self.id}, agent='{agent_name}', "
                f"preference='{self.social_preference}', style='{self.communication_style}', "
                f"reputation={self.reputation_score}, activity={self.activity_level})")

    def update_reputation(self, new_score: float):
        """Update reputation score with validation"""
        if not 0.0 <= new_score <= 100.0:
            raise ValueError(f"reputation_score must be between 0.0 and 100.0")

        self.reputation_score = new_score
        self.updated_at = datetime.utcnow()

    def update_counters(self, friends_delta: int = 0, interactions_delta: int = 0, communities_delta: int = 0):
        """Update social counters"""
        self.friends_count = max(0, self.friends_count + friends_delta)
        self.interactions_count = max(0, self.interactions_count + interactions_delta)
        self.communities_count = max(0, self.communities_count + communities_delta)
        self.updated_at = datetime.utcnow()

    def get_activity_score(self) -> float:
        """Calculate overall activity score based on various metrics"""
        # Normalize counters to 0-1 range (assuming reasonable maximums)
        friends_score = min(1.0, self.friends_count / 100.0)  # Max 100 friends
        interactions_score = min(1.0, self.interactions_count / 1000.0)  # Max 1000 interactions
        communities_score = min(1.0, self.communities_count / 10.0)  # Max 10 communities

        # Weighted average with activity level
        activity_score = (
            0.4 * self.activity_level +
            0.3 * friends_score +
            0.2 * interactions_score +
            0.1 * communities_score
        )

        return round(activity_score, 3)

    def get_social_profile_summary(self) -> dict:
        """Get comprehensive social profile summary"""
        return {
            'agent_name': self.agent.name if self.agent else f"Agent({self.agent_id})",
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'reputation_score': self.reputation_score,
            'activity_level': self.activity_level,
            'social_preference': self.social_preference,
            'communication_style': self.communication_style,
            'friends_count': self.friends_count,
            'interactions_count': self.interactions_count,
            'communities_count': self.communities_count,
            'activity_score': self.get_activity_score(),
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    def is_highly_active(self) -> bool:
        """Check if agent is highly active"""
        return self.activity_level >= 0.7

    def is_reputable(self) -> bool:
        """Check if agent has good reputation"""
        return self.reputation_score >= 70.0

    def is_socially_connected(self) -> bool:
        """Check if agent has good social connections"""
        return (self.friends_count >= 10 and
                self.interactions_count >= 50 and
                self.communities_count >= 2)