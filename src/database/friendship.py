"""
Friendship model for million-agent social platform
"""

from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship
from .config import Base


class Friendship(Base):
    """Friendship model representing social connections between agents"""

    __tablename__ = "friendships"

    id = Column(Integer, primary_key=True, index=True)
    initiator_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    recipient_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    # Friendship status and properties
    friendship_status = Column(String(20), nullable=False, default="pending", index=True)
    strength_level = Column(Float, nullable=False, default=0.5, index=True)

    # Interaction tracking
    interaction_count = Column(Integer, nullable=False, default=0)
    last_interaction = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    initiator = relationship("Agent", foreign_keys=[initiator_id], back_populates="initiated_friendships")
    recipient = relationship("Agent", foreign_keys=[recipient_id], back_populates="received_friendships")

    __table_args__ = (
        # Prevent duplicate friendships in same direction
        UniqueConstraint('initiator_id', 'recipient_id', name='uq_friendship_direction'),
        # Validate friendship status
        CheckConstraint("friendship_status IN ('pending', 'accepted', 'blocked', 'declined')", name='ck_friendship_status'),
        # Validate strength level range
        CheckConstraint('strength_level >= 0 AND strength_level <= 1', name='ck_strength_level_range'),
        # Validate interaction count
        CheckConstraint('interaction_count >= 0', name='ck_interaction_count_range'),
        # Prevent self-friendship
        CheckConstraint('initiator_id != recipient_id', name='ck_no_self_friendship'),
        {"sqlite_autoincrement": True}
    )

    def __init__(self, initiator_id: int, recipient_id: int, **kwargs):
        """Initialize friendship with validation"""
        self.initiator_id = initiator_id
        self.recipient_id = recipient_id

        # Prevent self-friendship
        if initiator_id == recipient_id:
            raise ValueError("Agent cannot be friends with themselves")

        # Validate friendship status
        valid_statuses = ["pending", "accepted", "blocked", "declined"]
        friendship_status = kwargs.get('friendship_status', 'pending')
        if friendship_status not in valid_statuses:
            raise ValueError(f"Invalid friendship_status: {friendship_status}. Must be one of: {valid_statuses}")
        self.friendship_status = friendship_status

        # Validate and set strength level
        strength_level = kwargs.get('strength_level', 0.5)
        if not 0.0 <= strength_level <= 1.0:
            raise ValueError(f"strength_level must be between 0.0 and 1.0")
        self.strength_level = strength_level

        # Set other attributes
        self.interaction_count = kwargs.get('interaction_count', 0)
        self.last_interaction = kwargs.get('last_interaction')

    def __str__(self):
        """String representation of friendship"""
        initiator_name = getattr(self.initiator, 'name', str(self.initiator_id))
        recipient_name = getattr(self.recipient, 'name', str(self.recipient_id))
        return f"Friendship({initiator_name} -> {recipient_name}, status='{self.friendship_status}', strength={self.strength_level})"

    def __repr__(self):
        """Detailed representation of friendship"""
        return (f"Friendship(id={self.id}, initiator={self.initiator_id}, "
                f"recipient={self.recipient_id}, status='{self.friendship_status}', "
                f"strength={self.strength_level}, interactions={self.interaction_count})")

    def update_status(self, new_status: str):
        """Update friendship status with validation"""
        valid_statuses = ["pending", "accepted", "blocked", "declined"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid friendship_status: {new_status}. Must be one of: {valid_statuses}")

        self.friendship_status = new_status
        self.updated_at = datetime.utcnow()

    def update_strength_level(self, new_level: float):
        """Update friendship strength level with validation"""
        if not 0.0 <= new_level <= 1.0:
            raise ValueError(f"strength_level must be between 0.0 and 1.0")

        self.strength_level = new_level
        self.updated_at = datetime.utcnow()

    def record_interaction(self):
        """Record a friendship interaction"""
        self.interaction_count += 1
        self.last_interaction = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if friendship is active (accepted)"""
        return self.friendship_status == "accepted"

    def is_strong(self, threshold: float = 0.7) -> bool:
        """Check if friendship is strong based on strength level"""
        return self.strength_level >= threshold

    def get_friendship_age(self) -> timedelta:
        """Get age of friendship"""
        return datetime.utcnow() - self.created_at

    def get_days_since_last_interaction(self) -> int:
        """Get days since last interaction"""
        if self.last_interaction is None:
            return (datetime.utcnow() - self.created_at).days
        return (datetime.utcnow() - self.last_interaction).days

    def is_stale(self, days: int = 30) -> bool:
        """Check if friendship is stale (no recent interaction)"""
        return self.get_days_since_last_interaction() > days

    def get_friendship_summary(self) -> dict:
        """Get comprehensive friendship summary"""
        return {
            'id': self.id,
            'initiator_id': self.initiator_id,
            'recipient_id': self.recipient_id,
            'friendship_status': self.friendship_status,
            'strength_level': self.strength_level,
            'is_active': self.is_active(),
            'is_strong': self.is_strong(),
            'interaction_count': self.interaction_count,
            'last_interaction': self.last_interaction,
            'days_since_last_interaction': self.get_days_since_last_interaction(),
            'is_stale': self.is_stale(),
            'friendship_age_days': self.get_friendship_age().days,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @staticmethod
    def get_valid_statuses() -> List[str]:
        """Get list of valid friendship statuses"""
        return ["pending", "accepted", "blocked", "declined"]

    @staticmethod
    def search_friendships_by_status(session, status: str, limit: int = 50) -> List['Friendship']:
        """Search friendships by status"""
        return session.query(Friendship).filter(
            Friendship.friendship_status == status
        ).limit(limit).all()

    @staticmethod
    def get_strong_friendships(session, threshold: float = 0.7, limit: int = 50) -> List['Friendship']:
        """Get strong friendships above threshold"""
        return session.query(Friendship).filter(
            Friendship.friendship_status == "accepted",
            Friendship.strength_level >= threshold
        ).limit(limit).all()

    @staticmethod
    def get_active_friendships_for_agent(session, agent_id: int, limit: int = 50) -> List['Friendship']:
        """Get active friendships for a specific agent"""
        return session.query(Friendship).filter(
            ((Friendship.initiator_id == agent_id) | (Friendship.recipient_id == agent_id)),
            Friendship.friendship_status == "accepted"
        ).limit(limit).all()

    @staticmethod
    def get_pending_friendships_for_agent(session, agent_id: int, limit: int = 50) -> List['Friendship']:
        """Get pending friendships for a specific agent (where they are recipient)"""
        return session.query(Friendship).filter(
            Friendship.recipient_id == agent_id,
            Friendship.friendship_status == "pending"
        ).limit(limit).all()

    @staticmethod
    def get_friendship_between_agents(session, agent1_id: int, agent2_id: int) -> Optional['Friendship']:
        """Get friendship between two agents (either direction)"""
        return session.query(Friendship).filter(
            ((Friendship.initiator_id == agent1_id) & (Friendship.recipient_id == agent2_id)) |
            ((Friendship.initiator_id == agent2_id) & (Friendship.recipient_id == agent1_id))
        ).first()

    def decay_strength(self, decay_rate: float = 0.1, max_decay: float = 0.3):
        """Apply decay to friendship strength based on inactivity"""
        if self.is_stale():
            decay_amount = min(decay_rate * self.get_days_since_last_interaction() / 30.0, max_decay)
            new_strength = max(0.1, self.strength_level - decay_amount)
            self.update_strength_level(new_strength)

    def strengthen_friendship(self, boost: float = 0.1):
        """Strengthen friendship based on positive interaction"""
        new_strength = min(1.0, self.strength_level + boost)
        self.update_strength_level(new_strength)
        self.record_interaction()