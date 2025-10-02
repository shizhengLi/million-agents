"""
Interaction model for million-agent social platform
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, CheckConstraint
from sqlalchemy.orm import relationship
from .config import Base


class Interaction(Base):
    """Interaction model representing social interactions between agents"""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    initiator_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    recipient_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    # Interaction details
    interaction_type = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=True)

    # Analysis metrics
    sentiment_score = Column(Float, nullable=True)  # -1.0 to 1.0
    engagement_score = Column(Float, nullable=True)  # 0.0 to 1.0

    # Interaction metadata (JSON for flexibility)
    interaction_metadata = Column(JSON, nullable=True)

    # Timestamps
    interaction_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    initiator = relationship("Agent", foreign_keys=[initiator_id], back_populates="interactions_as_initiator")
    recipient = relationship("Agent", foreign_keys=[recipient_id], back_populates="interactions_as_recipient")

    __table_args__ = (
        # Prevent self-interactions
        CheckConstraint('initiator_id != recipient_id', name='ck_interaction_no_self'),
        # Validate sentiment score range
        CheckConstraint('sentiment_score >= -1.0 AND sentiment_score <= 1.0', name='ck_sentiment_range'),
        # Validate engagement score range
        CheckConstraint('engagement_score >= 0.0 AND engagement_score <= 1.0', name='ck_engagement_range'),
        {"sqlite_autoincrement": True}
    )

    def __init__(self, initiator_id: int, recipient_id: int, interaction_type: str, **kwargs):
        """Initialize interaction with validation"""
        self.initiator_id = initiator_id
        self.recipient_id = recipient_id

        # Prevent self-interactions
        if initiator_id == recipient_id:
            raise ValueError("Cannot create self-interaction: initiator and recipient must be different")

        # Validate interaction type
        valid_types = [
            "conversation", "message", "collaboration", "sharing",
            "request", "response", "feedback", "introduction",
            "coordination", "negotiation", "support", "conflict"
        ]
        if interaction_type not in valid_types:
            raise ValueError(f"Invalid interaction_type: {interaction_type}. Must be one of: {valid_types}")
        self.interaction_type = interaction_type

        # Validate and set optional scores
        sentiment_score = kwargs.get('sentiment_score')
        if sentiment_score is not None:
            if not -1.0 <= sentiment_score <= 1.0:
                raise ValueError(f"sentiment_score must be between -1.0 and 1.0")
            self.sentiment_score = sentiment_score

        engagement_score = kwargs.get('engagement_score')
        if engagement_score is not None:
            if not 0.0 <= engagement_score <= 1.0:
                raise ValueError(f"engagement_score must be between 0.0 and 1.0")
            self.engagement_score = engagement_score

        # Set other attributes
        self.content = kwargs.get('content')
        self.interaction_metadata = kwargs.get('interaction_metadata') or kwargs.get('metadata')

        # Set interaction_time if provided
        interaction_time = kwargs.get('interaction_time')
        if interaction_time:
            self.interaction_time = interaction_time

    def __str__(self):
        """String representation of interaction"""
        initiator_name = self.initiator.name if self.initiator else f"Agent({self.initiator_id})"
        recipient_name = self.recipient.name if self.recipient else f"Agent({self.recipient_id})"
        return f"Interaction({initiator_name} -> {recipient_name}, type='{self.interaction_type}')"

    def __repr__(self):
        """Detailed representation of interaction"""
        initiator_name = self.initiator.name if self.initiator else f"Agent({self.initiator_id})"
        recipient_name = self.recipient.name if self.recipient else f"Agent({self.recipient_id})"
        return (f"Interaction(id={self.id}, initiator='{initiator_name}', "
                f"recipient='{recipient_name}', type='{self.interaction_type}', "
                f"sentiment={self.sentiment_score}, engagement={self.engagement_score})")

    def get_sentiment_label(self) -> str:
        """Get sentiment label based on sentiment score"""
        if self.sentiment_score is None:
            return "neutral"

        if self.sentiment_score > 0.3:
            return "positive"
        elif self.sentiment_score < -0.3:
            return "negative"
        else:
            return "neutral"

    def get_engagement_level(self) -> str:
        """Get engagement level based on engagement score"""
        if self.engagement_score is None:
            return "medium"

        if self.engagement_score >= 0.7:
            return "high"
        elif self.engagement_score >= 0.4:
            return "medium"
        else:
            return "low"

    def update_analysis_scores(self, sentiment: float = None, engagement: float = None):
        """Update sentiment and engagement scores with validation"""
        if sentiment is not None:
            if not -1.0 <= sentiment <= 1.0:
                raise ValueError(f"sentiment_score must be between -1.0 and 1.0")
            self.sentiment_score = sentiment

        if engagement is not None:
            if not 0.0 <= engagement <= 1.0:
                raise ValueError(f"engagement_score must be between 0.0 and 1.0")
            self.engagement_score = engagement

    def add_metadata(self, key: str, value):
        """Add or update metadata field"""
        if self.interaction_metadata is None:
            self.interaction_metadata = {}
        self.interaction_metadata[key] = value

    def get_metadata_field(self, key: str, default=None):
        """Get metadata field with default value"""
        if self.interaction_metadata is None:
            return default
        return self.interaction_metadata.get(key, default)

    def is_high_engagement(self) -> bool:
        """Check if interaction has high engagement"""
        return self.engagement_score is not None and self.engagement_score >= 0.7

    def is_positive_sentiment(self) -> bool:
        """Check if interaction has positive sentiment"""
        return self.sentiment_score is not None and self.sentiment_score > 0.3

    def get_interaction_summary(self) -> dict:
        """Get comprehensive interaction summary"""
        return {
            'id': self.id,
            'initiator_id': self.initiator_id,
            'recipient_id': self.recipient_id,
            'initiator_name': self.initiator.name if self.initiator else None,
            'recipient_name': self.recipient.name if self.recipient else None,
            'interaction_type': self.interaction_type,
            'content': self.content,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.get_sentiment_label(),
            'engagement_score': self.engagement_score,
            'engagement_level': self.get_engagement_level(),
            'metadata': self.interaction_metadata,
            'interaction_time': self.interaction_time,
            'created_at': self.created_at,
            'is_high_engagement': self.is_high_engagement(),
            'is_positive_sentiment': self.is_positive_sentiment()
        }

    @staticmethod
    def get_interaction_types() -> list:
        """Get list of valid interaction types"""
        return [
            "conversation", "message", "collaboration", "sharing",
            "request", "response", "feedback", "introduction",
            "coordination", "negotiation", "support", "conflict"
        ]

    def is_mutual_interaction(self, other_interaction) -> bool:
        """Check if this is a mutual/reverse interaction with another"""
        if not isinstance(other_interaction, Interaction):
            return False

        return (
            self.initiator_id == other_interaction.recipient_id and
            self.recipient_id == other_interaction.initiator_id and
            self.interaction_type == other_interaction.interaction_type
        )